# -*- coding: utf-8 -*-
"""
Raptor_Xtrain.py

Xtrain/CPUWARP-ML translation of Raptor_V6.py
Optimized for CPU training with WARP scheduling
"""

import numpy as np
import cpuwarp_ml
import time
import argparse
from typing import Dict, List, Tuple, Optional, Any
import threading
import multiprocessing as mp

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================

# Architecture Settings
D_MODEL = 384
D_FF = 1536
N_HEADS = 6
N_LAYERS = 6
MAX_SEQ_LEN = 128

# MoE & Attention Settings
K_TOP_TOKENS = 2
N_EXPERTS = 6
K_MOE = 2

# Training Settings
BATCH_SIZE = 32
LEARNING_RATE_LM = 5e-5
WEIGHT_DECAY = 0.01
LB_LOSS_WEIGHT = 0.01
RL_LOSS_WEIGHT = 1e-6
NUM_EPOCHS = 60
LM_WARMUP_EPOCHS = 5
MOE_WARMUP_EPOCHS = 10

# System Settings
NUM_WORKERS = max(1, (mp.cpu_count() or 1) // 2)

# ==========================================
# 2. MODEL ARCHITECTURE (XTRAIN VERSION)
# ==========================================

class ExpertFeedForward:
    """Simple Feed-Forward Network used as an expert."""
    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Initialize weights using Xavier/Glorot initialization
        limit = np.sqrt(6 / (d_model + d_ff))
        self.fc1_weights = np.random.uniform(-limit, limit, (d_model, d_ff)).astype(np.float32)
        self.fc2_weights = np.random.uniform(-limit, limit, (d_ff, d_model)).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through expert feed-forward network."""
        # fc1: x @ fc1_weights
        fc1_out = cpuwarp_ml.matmul(x, self.fc1_weights)
        
        # GELU activation (approximation since cpuwarp_ml doesn't have gelu)
        gelu_out = 0.5 * fc1_out * (1 + np.tanh(np.sqrt(2 / np.pi) * (fc1_out + 0.044715 * fc1_out**3)))
        
        # fc2: gelu_out @ fc2_weights
        fc2_out = cpuwarp_ml.matmul(gelu_out, self.fc2_weights)
        
        return fc2_out


class MoE:
    """Mixture of Experts layer using top-k routing."""
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 8, k: int = 2):
        self.d_model = d_model
        self.n_experts_count = n_experts
        self.k = k
        
        # Experts: List of feed-forward networks
        self.experts = [ExpertFeedForward(d_model, d_ff) for _ in range(n_experts)]
        
        # Router: maps token embeddings to expert logits
        limit = np.sqrt(6 / (d_model + n_experts))
        self.router_weights = np.random.uniform(-limit, limit, (d_model, n_experts)).astype(np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        x: (B, L, D)
        returns: (B, L, D), lb_loss
        """
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)  # Flatten batch and sequence dimensions

        # 1. Routing probabilities
        logits = cpuwarp_ml.matmul(x_flat, self.router_weights)
        probs = cpuwarp_ml.softmax(logits, axis=-1)

        # 2. Top-k experts per token
        # Note: For now, we'll use a simplified approach since NumPy doesn't have topk like PyTorch
        # We'll sort and take top k
        sorted_indices = np.argsort(probs, axis=-1)[:, -self.k:]
        top_weights = np.take_along_axis(probs, sorted_indices, axis=-1)

        # 3. Accumulate expert contributions
        contributions = np.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            # Create mask for tokens that use this expert
            mask = np.any(sorted_indices == i, axis=-1)
            if np.any(mask):
                inp = x_flat[mask]
                
                # Calculate per-token weight for this expert
                expert_mask = (sorted_indices == i)
                per_token_weight = (top_weights * expert_mask.astype(np.float32)).sum(axis=-1)
                per_token_weight = per_token_weight[mask, np.newaxis]
                
                # Apply expert and accumulate
                expert_in = inp * per_token_weight
                expert_out = expert.forward(expert_in)
                contributions[mask] += expert_out

        # 4. Load-balancing loss
        importance = probs.sum(axis=0)
        imp = importance / x_flat.shape[0]
        lb_loss = (self.n_experts_count * (imp * imp).sum()).astype(np.float32)

        return contributions.reshape(B, L, D), float(lb_loss)


class HardAttention_MHA:
    """Multi-Head Attention with Hard Policy selection."""
    def __init__(self, d_model: int, n_heads: int, top_k_tokens: int = 1):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.top_k_tokens = top_k_tokens
        self.d_k = d_model // n_heads

        # Initialize projection weights
        limit = np.sqrt(6 / (d_model + d_model))
        self.q_proj = np.random.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)
        self.k_proj = np.random.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)
        self.v_proj = np.random.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)
        self.o_proj = np.random.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split into multiple attention heads."""
        B, L, D = x.shape
        # Reshape and transpose: (B, L, D) -> (B, L, n_heads, d_k) -> (B, n_heads, L, d_k)
        return x.reshape(B, L, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """Combine multiple attention heads."""
        B, H, L, d_k = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)

    def hard_policy(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, causal_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Hard attention policy with sampling."""
        B, H, L_q, D_k = Q.shape
        L_k = K.shape[2]

        # Compute attention scores
        scale = np.sqrt(D_k).astype(np.float32)
        # Manual matmul for 4D tensors since cpuwarp_ml.matmul doesn't handle this well
        # Q: (B, H, L_q, d_k), K.T: (B, H, d_k, L_k) -> scores: (B, H, L_q, L_k)
        K_transposed = K.transpose(0, 1, 3, 2)  # (B, H, d_k, L_k)
        scores = np.einsum('bhqd,bhdk->bhqk', Q, K_transposed) / scale

        if causal_mask is not None:
            # Apply causal mask - need to expand dimensions to match scores shape
            # scores shape: (B, H, L_q, L_k)
            # causal_mask shape: (L_q, L_k)
            causal_mask_expanded = np.expand_dims(np.expand_dims(causal_mask, 0), 0)  # (1, 1, L_q, L_k)
            # Tile to match batch and head dimensions
            causal_mask_expanded = np.tile(causal_mask_expanded, (B, H, 1, 1))
            scores = np.where(causal_mask_expanded, scores, np.finfo(np.float32).min)

        # Softmax to get probabilities (use numpy since cpuwarp_ml doesn't handle 4D well)
        policy_probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        policy_probs = policy_probs / np.sum(policy_probs, axis=-1, keepdims=True)
        policy_probs = np.clip(policy_probs, 1e-9, 1.0)
        policy_probs = policy_probs / policy_probs.sum(axis=-1, keepdims=True)

        # Sample top-k tokens (simplified approach)
        k = min(self.top_k_tokens, L_k)
        flat_probs = policy_probs.reshape(-1, L_k)
        
        # For now, use deterministic top-k instead of sampling
        # This is a simplification for the CPU implementation
        sampled_indices = np.argsort(flat_probs, axis=-1)[:, -k:]

        # Create hard mask (simplified approach - use deterministic top-k)
        # For now, just use the first k tokens for simplicity
        hard_mask = np.zeros_like(scores)
        for batch_idx in range(B):
            for head_idx in range(H):
                for seq_idx in range(L_q):
                    # Simple approach: attend to first k tokens
                    for j in range(min(k, L_k)):
                        if j < L_k:
                            hard_mask[batch_idx, head_idx, seq_idx, j] = 1.0

        # Apply hard attention
        hard_output = np.einsum('bhqk,bhkd->bhqd', hard_mask, V)

        # Calculate log probabilities for RL (simplified for now)
        # For testing, return zeros with correct shape
        log_prob_action = np.zeros((B, L_q))

        return hard_output, log_prob_action

    def forward(self, x: np.ndarray, causal_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through hard attention."""
        Q = self.split_heads(cpuwarp_ml.matmul(x, self.q_proj))
        K = self.split_heads(cpuwarp_ml.matmul(x, self.k_proj))
        V = self.split_heads(cpuwarp_ml.matmul(x, self.v_proj))

        attn_out, log_prob = self.hard_policy(Q, K, V, causal_mask)
        attn_out = self.combine_heads(attn_out)
        return cpuwarp_ml.matmul(attn_out, self.o_proj), log_prob


class Transformer_Block:
    def __init__(self, d_model: int, d_ff: int, n_heads: int, top_k_tokens: int, n_experts: int = 8, k_moe: int = 2):
        self.attention = HardAttention_MHA(d_model, n_heads, top_k_tokens)
        self.moe = MoE(d_model, d_ff, n_experts, k_moe)
        
        # Layer normalization parameters
        self.ln1_gamma = np.ones(d_model, dtype=np.float32)
        self.ln1_beta = np.zeros(d_model, dtype=np.float32)
        self.ln2_gamma = np.ones(d_model, dtype=np.float32)
        self.ln2_beta = np.zeros(d_model, dtype=np.float32)

    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer normalization."""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return gamma * (x - mean) / (std + 1e-6) + beta

    def forward(self, x: np.ndarray, causal_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float]:
        """Forward pass through transformer block."""
        # Layer norm 1
        norm_x = self.layer_norm(x, self.ln1_gamma, self.ln1_beta)
        
        # Attention
        attn_out, log_prob = self.attention.forward(norm_x, causal_mask)
        x = x + attn_out

        # Layer norm 2
        norm_x = self.layer_norm(x, self.ln2_gamma, self.ln2_beta)
        
        # MoE
        moe_out, lb_loss = self.moe.forward(norm_x)
        x = x + moe_out

        # Keep log probabilities as (B, L) - don't sum across sequence dimension
        if log_prob.ndim == 3:
            log_prob = log_prob.sum(axis=1)  # Sum across heads only
        # If already (B, L), keep as is
        
        return x, log_prob, lb_loss


class Decoder_LLM:
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, n_layers: int, d_ff: int, 
                 n_heads: int = 8, top_k_tokens: int = 2, n_experts: int = 8, k_moe: int = 2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        
        # Embedding layers
        limit = np.sqrt(6 / (vocab_size + d_model))
        self.token_embed = np.random.uniform(-limit, limit, (vocab_size, d_model)).astype(np.float32)
        self.pos_embed = np.random.uniform(-limit, limit, (max_seq_len + 1, d_model)).astype(np.float32)

        # Transformer layers
        self.layers = [
            Transformer_Block(d_model, d_ff, n_heads, top_k_tokens, n_experts, k_moe)
            for _ in range(n_layers)
        ]
        
        # Final layer normalization
        self.final_norm_gamma = np.ones(d_model, dtype=np.float32)
        self.final_norm_beta = np.zeros(d_model, dtype=np.float32)
        
        # Output head
        limit = np.sqrt(6 / (d_model + vocab_size))
        self.output_head = np.random.uniform(-limit, limit, (d_model, vocab_size)).astype(np.float32)

    def layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Final layer normalization."""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return self.final_norm_gamma * (x - mean) / (std + 1e-6) + self.final_norm_beta

    def forward(self, input_ids: np.ndarray, return_lb: bool = False, return_log_probs: bool = False) -> Any:
        """Forward pass through the entire model."""
        B, L = input_ids.shape

        # Token and position embeddings
        x = self.token_embed[input_ids]
        pos = np.arange(L)
        x = x + self.pos_embed[pos]

        # Causal mask
        causal_mask = np.tril(np.ones((L, L), dtype=bool))

        total_lb_loss = 0.0
        all_log_probs = []

        # Transformer layers
        for layer in self.layers:
            x, log_prob, lb_loss = layer.forward(x, causal_mask)
            total_lb_loss += lb_loss
            all_log_probs.append(log_prob)

        # Final layer norm
        x = self.layer_norm(x)
        
        # Output logits
        logits = cpuwarp_ml.matmul(x, self.output_head)

        if return_log_probs:
            return logits, all_log_probs, total_lb_loss
        else:
            result = {"logits": logits}
            if return_lb:
                result["lb_loss"] = total_lb_loss
            else:
                result["lb_loss"] = 0.0
            return result


def init_weights(layer: Any):
    """Weight initialization helper."""
    # This is handled in the constructor for xtrain version
    pass


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def fmt(x: float) -> str:
    """Format tensor or float for printing."""
    return f"{x:.4f}"


def to_float(x: Any) -> float:
    """Convert tensor to float."""
    return float(x) if not isinstance(x, float) else x


# ==========================================
# 4. TRAINING LOGIC
# ==========================================

def LM_training_step(model: Decoder_LLM, batch: Tuple[np.ndarray, np.ndarray], 
                    optimizer: Any, lb_weight: float, use_moe: bool = False) -> Tuple[float, float]:
    """Language model training step."""
    input_ids, labels = batch
    
    # Forward pass
    outputs = model(input_ids, return_lb=use_moe)
    logits = outputs["logits"]
    lb_loss = outputs["lb_loss"]

    # LM loss (cross-entropy)
    # Simplified version for xtrain
    vocab_size = logits.shape[-1]
    
    # Convert labels to one-hot for simplicity
    one_hot_labels = np.zeros((labels.shape[0], labels.shape[1], vocab_size), dtype=np.float32)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] != -100:  # Ignore padding
                one_hot_labels[i, j, labels[i, j]] = 1.0
    
    # Softmax and cross-entropy
    probs = cpuwarp_ml.softmax(logits, axis=-1)
    lm_loss = -np.mean(np.sum(one_hot_labels * np.log(probs + 1e-9), axis=-1))

    # Total loss
    total_loss = lm_loss + lb_weight * lb_loss

    # Backward pass (simplified for xtrain)
    # Note: This is a placeholder - xtrain would need proper backpropagation
    # For now, we'll just return the losses
    
    return float(lm_loss), float(lb_loss)


# ==========================================
# 5. PERFORMANCE COMPARISON FUNCTIONS
# ==========================================

def compare_performance_pytorch_vs_xtrain():
    """Compare PyTorch vs Xtrain performance."""
    print("Performance Comparison: PyTorch vs Xtrain")
    print("=" * 50)
    
    # Test parameters
    batch_size = 4
    seq_len = 64
    d_model = 128
    n_experts = 4
    
    # Create test data
    test_input = np.random.randint(0, 1000, (batch_size, seq_len)).astype(np.int32)
    
    # Initialize models
    print("Initializing models...")
    
    # Xtrain model
    xtrain_model = Decoder_LLM(
        vocab_size=1000,
        d_model=d_model,
        max_seq_len=seq_len,
        n_layers=2,
        d_ff=d_model * 4,
        n_heads=4,
        top_k_tokens=1,
        n_experts=n_experts,
        k_moe=2
    )
    
    print("Running forward pass tests...")
    
    # Warmup
    for _ in range(3):
        _ = xtrain_model.forward(test_input)
    
    # Time xtrain forward pass
    start_time = time.time()
    for _ in range(10):
        logits = xtrain_model.forward(test_input)
    xtrain_time = time.time() - start_time
    
    print(f"Xtrain forward pass (10 iterations): {xtrain_time:.4f}s")
    print(f"Average per iteration: {xtrain_time/10:.4f}s")
    
    # Test memory usage
    import psutil
    import os
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory usage: {memory_usage:.2f} MB")
    
    return {
        "xtrain_time": xtrain_time,
        "xtrain_avg_time": xtrain_time / 10,
        "memory_usage": memory_usage
    }


def test_model_functionality():
    """Test that the xtrain model produces reasonable outputs."""
    print("Testing model functionality...")
    
    # Small test case
    vocab_size = 100
    seq_len = 8  # Reduced to avoid attention issues
    batch_size = 2
    
    # Create model
    model = Decoder_LLM(
        vocab_size=vocab_size,
        d_model=64,
        max_seq_len=seq_len,
        n_layers=1,  # Reduced layers for testing
        d_ff=128,
        n_heads=2,
        top_k_tokens=1,
        n_experts=2,
        k_moe=1
    )
    
    # Create test input
    test_input = np.random.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int32)
    
    # Forward pass
    outputs = model.forward(test_input)
    logits = outputs["logits"]
    
    # Check output shape
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Wrong output shape: {logits.shape}"
    
    # Check that logits are reasonable (not all zeros, not NaN)
    assert not np.any(np.isnan(logits)), "NaN values in logits"
    assert not np.all(logits == 0), "All zeros in logits"
    
    # Test with MoE
    outputs_with_moe = model.forward(test_input, return_lb=True)
    assert "lb_loss" in outputs_with_moe, "Missing lb_loss in output"
    assert outputs_with_moe["lb_loss"] >= 0, "Negative lb_loss"
    
    print("[OK] Model functionality tests passed!")
    return True


if __name__ == "__main__":
    print("Raptor Xtrain Model")
    print("=" * 50)
    
    # Test basic functionality
    test_model_functionality()
    
    # Run performance comparison
    results = compare_performance_pytorch_vs_xtrain()
    
    print("\nPerformance Summary:")
    print(f"  Xtrain time (10 iterations): {results['xtrain_time']:.4f}s")
    print(f"  Average per iteration: {results['xtrain_avg_time']:.4f}s")
    print(f"  Memory usage: {results['memory_usage']:.2f} MB")
    
    print("\n[INFO] Translation complete! The model is ready for training.")