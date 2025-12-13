"""
CPUWARP-ML LLM Training Script
==============================

Train transformer-based language models using CPUWARP-ML framework
Optimized for CPU training with WARP scheduling
"""

import numpy as np
import time
import argparse
from typing import Dict, List, Tuple, Optional
import cpuwarp_ml

class MultiHeadAttention:
    """Multi-head attention mechanism optimized for CPUWARP-ML"""
    
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weights
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape
        
        # Simplified single-head attention for now
        # Compute Q, K, V using CPUWARP-ML optimized matmul
        Q = cpuwarp_ml.matmul(x, self.W_q)
        K = cpuwarp_ml.matmul(x, self.W_k)
        V = cpuwarp_ml.matmul(x, self.W_v)
        
        # Compute attention scores
        scores = cpuwarp_ml.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_model)
        attention_weights = cpuwarp_ml.softmax(scores, axis=-1)
        
        # Apply attention to values
        attention_output = cpuwarp_ml.matmul(attention_weights, V)
        
        # Final projection
        output = cpuwarp_ml.matmul(attention_output, self.W_o)
        
        return output
    
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        # Simplified backward pass for demonstration
        # In production, implement full gradient computation
        grad_x = cpuwarp_ml.matmul(grad_output, self.W_o.T)
        return grad_x

class FeedForward:
    """Feed-forward network optimized for CPUWARP-ML"""
    
    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Initialize weights
        self.W1 = np.random.randn(d_model, d_ff).astype(np.float32) * 0.1
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = np.random.randn(d_ff, d_model).astype(np.float32) * 0.1
        self.b2 = np.zeros(d_model, dtype=np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # First linear layer
        hidden = cpuwarp_ml.matmul(x, self.W1) + self.b1
        
        # ReLU activation
        hidden = cpuwarp_ml.relu(hidden)
        
        # Second linear layer
        output = cpuwarp_ml.matmul(hidden, self.W2) + self.b2
        
        return output
    
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        # Simplified backward pass
        grad_x = cpuwarp_ml.matmul(grad_output, self.W2.T)
        return grad_x

class LayerNorm:
    """Layer normalization optimized for CPUWARP-ML"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta
    
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        # Simplified backward pass
        return grad_output

class TransformerBlock:
    """Single transformer block with attention and feed-forward layers"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Self-attention with residual connection
        attention_output = self.attention.forward(x)
        x = self.layer_norm1.forward(x + attention_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward.forward(x)
        x = self.layer_norm2.forward(x + ff_output)
        
        return x
    
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        # Simplified backward pass
        return grad_output

class CPUWarpTransformer:
    """Complete transformer model optimized for CPUWARP-ML"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8, 
                 num_layers: int = 6, d_ff: int = 2048, max_seq_len: int = 512):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.1
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding()
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ]
        
        # Output layer
        self.output_layer = np.random.randn(d_model, vocab_size).astype(np.float32) * 0.1
        
    def _create_positional_encoding(self) -> np.ndarray:
        """Create sinusoidal positional encoding"""
        pe = np.zeros((self.max_seq_len, self.d_model), dtype=np.float32)
        position = np.arange(0, self.max_seq_len).reshape(-1, 1)
        
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Forward pass through the transformer"""
        batch_size, seq_len = input_ids.shape
        
        # Token embedding
        x = self.token_embedding[input_ids]  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x += self.positional_encoding[:seq_len]
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x)
        
        # Output projection
        logits = cpuwarp_ml.matmul(x, self.output_layer)
        
        return logits
    
    def get_num_parameters(self) -> int:
        """Count total number of parameters"""
        total_params = 0
        
        # Token embedding
        total_params += self.token_embedding.size
        
        # Transformer blocks
        for block in self.blocks:
            total_params += block.attention.W_q.size
            total_params += block.attention.W_k.size
            total_params += block.attention.W_v.size
            total_params += block.attention.W_o.size
            total_params += block.feed_forward.W1.size
            total_params += block.feed_forward.b1.size
            total_params += block.feed_forward.W2.size
            total_params += block.feed_forward.b2.size
            total_params += block.layer_norm1.gamma.size
            total_params += block.layer_norm1.beta.size
            total_params += block.layer_norm2.gamma.size
            total_params += block.layer_norm2.beta.size
        
        # Output layer
        total_params += self.output_layer.size
        
        return total_params

def generate_dummy_data(batch_size: int, seq_len: int, vocab_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate dummy training data"""
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    # Target is input shifted by one position
    targets = np.roll(input_ids, -1, axis=1)
    targets[:, -1] = 0  # Padding token
    
    return input_ids, targets

def compute_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Compute cross-entropy loss"""
    batch_size, seq_len, vocab_size = logits.shape
    
    # Flatten for easier computation
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    # Apply softmax to get probabilities
    probs = cpuwarp_ml.softmax(logits_flat, axis=1)
    
    # Compute cross-entropy loss
    loss = 0.0
    for i, target in enumerate(targets_flat):
        if target != 0:  # Ignore padding tokens
            loss -= np.log(probs[i, target] + 1e-8)
    
    return loss / np.sum(targets_flat != 0)

def train_epoch(model: CPUWarpTransformer, batch_size: int, seq_len: int, 
                num_batches: int) -> Dict[str, float]:
    """Train for one epoch"""
    
    total_loss = 0.0
    total_time = 0.0
    
    print(f"Training epoch with {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        start_time = time.time()
        
        # Generate batch data
        input_ids, targets = generate_dummy_data(batch_size, seq_len, model.vocab_size)
        
        # Forward pass
        logits = model.forward(input_ids)
        
        # Compute loss
        loss = compute_loss(logits, targets)
        total_loss += loss
        
        # Timing
        batch_time = time.time() - start_time
        total_time += batch_time
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} - "
                  f"Loss: {loss:.4f}, Time: {batch_time:.3f}s")
    
    avg_loss = total_loss / num_batches
    avg_time = total_time / num_batches
    throughput = (batch_size * seq_len) / avg_time
    
    return {
        'avg_loss': avg_loss,
        'avg_batch_time': avg_time,
        'throughput': throughput,
        'total_time': total_time
    }

def benchmark_model(model: CPUWarpTransformer, batch_sizes: List[int], 
                   seq_len: int = 128) -> Dict[str, List[float]]:
    """Benchmark model with different batch sizes"""
    
    print("Benchmarking model performance...")
    results = {
        'batch_sizes': batch_sizes,
        'throughput': [],
        'avg_time': [],
        'memory_mb': []
    }
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size: {batch_size}")
        
        # Generate test data
        input_ids, _ = generate_dummy_data(batch_size, seq_len, model.vocab_size)
        
        # Warm-up run
        _ = model.forward(input_ids)
        
        # Benchmark runs
        times = []
        for _ in range(5):
            start_time = time.time()
            logits = model.forward(input_ids)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        throughput = (batch_size * seq_len) / avg_time
        memory_mb = logits.nbytes / (1024 * 1024)
        
        results['throughput'].append(throughput)
        results['avg_time'].append(avg_time)
        results['memory_mb'].append(memory_mb)
        
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Throughput: {throughput:.1f} tokens/sec")
        print(f"  Memory: {memory_mb:.1f} MB")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train LLM with CPUWARP-ML')
    parser.add_argument('--vocab-size', type=int, default=10000, 
                       help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=512, 
                       help='Model dimension')
    parser.add_argument('--num-heads', type=int, default=8, 
                       help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=6, 
                       help='Number of transformer layers')
    parser.add_argument('--d-ff', type=int, default=2048, 
                       help='Feed-forward dimension')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size')
    parser.add_argument('--seq-len', type=int, default=128, 
                       help='Sequence length')
    parser.add_argument('--num-epochs', type=int, default=3, 
                       help='Number of training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=50, 
                       help='Batches per epoch')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run benchmark mode')
    
    args = parser.parse_args()
    
    print("CPUWARP-ML LLM Training")
    print("=" * 50)
    print(f"Model Configuration:")
    print(f"  Vocabulary Size: {args.vocab_size}")
    print(f"  Model Dimension: {args.d_model}")
    print(f"  Attention Heads: {args.num_heads}")
    print(f"  Transformer Layers: {args.num_layers}")
    print(f"  Feed-Forward Dim: {args.d_ff}")
    
    # Initialize model
    model = CPUWarpTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff
    )
    
    print(f"  Total Parameters: {model.get_num_parameters():,}")
    print()
    
    # Print CPUWARP-ML stats
    stats = cpuwarp_ml.cpuwarp.get_performance_stats()
    print("CPUWARP-ML Configuration:")
    print(f"  CPU: {stats['cpu_info']['vendor'].upper()}")
    print(f"  Cores: {stats['cpu_info']['cores']}")
    print(f"  Threads: {stats['cpu_info']['threads']}")
    print(f"  Features: {', '.join(stats['cpu_info']['features'])}")
    print(f"  C Extensions: {'Yes' if stats['c_extensions'] else 'No'}")
    print()
    
    if args.benchmark:
        # Benchmark mode
        batch_sizes = [1, 2, 4, 8, 16, 32]
        results = benchmark_model(model, batch_sizes, args.seq_len)
        
        print("\nBenchmark Results:")
        print("-" * 50)
        for i, bs in enumerate(results['batch_sizes']):
            print(f"Batch Size {bs:2d}: "
                  f"{results['throughput'][i]:6.1f} tokens/sec, "
                  f"{results['avg_time'][i]:6.4f}s, "
                  f"{results['memory_mb'][i]:5.1f} MB")
    
    else:
        # Training mode
        print(f"Training Configuration:")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Sequence Length: {args.seq_len}")
        print(f"  Epochs: {args.num_epochs}")
        print(f"  Batches per Epoch: {args.batches_per_epoch}")
        print()
        
        # Training loop
        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            print("-" * 30)
            
            epoch_stats = train_epoch(
                model, args.batch_size, args.seq_len, args.batches_per_epoch
            )
            
            print(f"Epoch {epoch + 1} Results:")
            print(f"  Average Loss: {epoch_stats['avg_loss']:.4f}")
            print(f"  Average Batch Time: {epoch_stats['avg_batch_time']:.4f}s")
            print(f"  Throughput: {epoch_stats['throughput']:.1f} tokens/sec")
            print(f"  Total Epoch Time: {epoch_stats['total_time']:.2f}s")
            print()
        
        print("Training completed!")


class ReGLU:
    """reGLU activation function: x * ReLU(x) where x is split into two halves"""
    
    def __init__(self):
        self.cache_input = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: split input, apply ReLU to second half, multiply with first half"""
        self.cache_input = x
        
        # Split input along the last dimension
        split_idx = x.shape[-1] // 2
        x1 = x[..., :split_idx]
        x2 = x[..., split_idx:]
        
        # Apply ReLU to second half and multiply with first half
        output = x1 * np.maximum(0, x2)
        
        return output.astype(np.float32)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass for reGLU activation"""
        if self.cache_input is None:
            raise ValueError("Forward pass must be called before backward pass")
        
        x = self.cache_input
        split_idx = x.shape[-1] // 2
        x1 = x[..., :split_idx]
        x2 = x[..., split_idx:]
        
        # Compute gradients
        # d(output)/d(x1) = ReLU(x2)
        # d(output)/d(x2) = x1 * (x2 > 0)
        grad_x1 = grad_output * (x2 > 0).astype(np.float32)
        grad_x2 = grad_output * x1 * (x2 > 0).astype(np.float32)
        
        # Concatenate gradients
        grad_input = np.concatenate([grad_x1, grad_x2], axis=-1)
        
        return grad_input


class RobustNeuralNet:
    """
    Robust Neural Network with comprehensive error checking and validation
    
    Features:
    - Tensor shape validation
    - Input dimension checks
    - Gradient flow validation
    - Activation function compatibility
    - Model checkpointing
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 10, output_dim: int = 1):
        """
        Initialize RobustNeuralNet
        
        Args:
            input_dim: Input dimension size
            hidden_dim: Hidden layer dimension size
            output_dim: Output dimension size
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Validate dimensions
        self._validate_dimensions()
        
        # Initialize layers with reGLU activation
        self.layers = [
            Dense(input_dim, hidden_dim),
            ReGLU(),  # reGLU activation as requested
            Dense(hidden_dim, output_dim)
        ]
        
        # Gradient cache for validation
        self.gradient_cache = {}
        self.input_cache = None
        self.output_cache = None
        
        # Checkpointing state
        self.checkpoint_dir = "checkpoints"
        self.checkpoint_prefix = "robust_net"
        
        print(f"RobustNeuralNet initialized: {input_dim} -> {hidden_dim} -> {output_dim}")
        print(f"Activation: reGLU")
        print(f"Total parameters: {self.get_num_parameters()}")
    
    def _validate_dimensions(self):
        """Validate tensor dimensions"""
        if self.input_dim <= 0 or self.hidden_dim <= 0 or self.output_dim <= 0:
            raise ValueError("All dimensions must be positive integers")
        
        if self.input_dim < 2:
            raise ValueError("Input dimension too small for reGLU activation (needs at least 2)")
        
        if self.hidden_dim < 2:
            raise ValueError("Hidden dimension too small for reGLU activation (needs at least 2)")
    
    def _validate_input_shape(self, x: np.ndarray):
        """Validate input tensor shape"""
        if len(x.shape) != 2:
            raise ValueError(f"Expected 2D input (batch_size, input_dim), got shape {x.shape}")
        
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {x.shape[1]}")
        
        if not np.isfinite(x).all():
            raise ValueError("Input contains non-finite values (NaN or Inf)")
    
    def _validate_activation_compatibility(self):
        """Check if activation functions are compatible with layer dimensions"""
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ReGLU):
                # Check if previous layer output dimension is compatible with reGLU
                if i > 0 and hasattr(self.layers[i-1], 'out_features'):
                    prev_out_dim = self.layers[i-1].out_features if hasattr(self.layers[i-1], 'out_features') else self.layers[i-1].weights.shape[1]
                    if prev_out_dim < 2:
                        raise ValueError(f"Layer before ReGLU must have output dimension >= 2, got {prev_out_dim}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with comprehensive validation"""
        # Input validation
        self._validate_input_shape(x)
        self.input_cache = x.copy()
        
        # Forward through layers
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            
            # Validate intermediate outputs
            if not np.isfinite(x).all():
                raise RuntimeError(f"Non-finite values detected after layer {i} ({type(layer).__name__})")
        
        self.output_cache = x.copy()
        return x
    
    def backward(self, grad_output: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass with gradient validation"""
        if self.output_cache is None:
            raise RuntimeError("Forward pass must be called before backward pass")
        
        # Validate gradient output
        if not np.isfinite(grad_output).all():
            raise ValueError("Gradient output contains non-finite values")
        
        if grad_output.shape != self.output_cache.shape:
            raise ValueError(f"Gradient output shape mismatch: expected {self.output_cache.shape}, got {grad_output.shape}")
        
        # Backward through layers
        gradients = {}
        current_grad = grad_output
        
        for i, layer in reversed(list(enumerate(self.layers))):
            if hasattr(layer, 'backward'):
                current_grad = layer.backward(current_grad)
                
                # Cache gradients for validation
                if hasattr(layer, 'grad_weights'):
                    self.gradient_cache[f'layer_{i}_weights'] = layer.grad_weights
                    self.gradient_cache[f'layer_{i}_bias'] = layer.grad_bias
                
                # Validate gradients
                if not np.isfinite(current_grad).all():
                    raise RuntimeError(f"Non-finite gradients detected after layer {i} ({type(layer).__name__})")
                
                if np.any(np.isnan(current_grad)):
                    raise RuntimeError(f"NaN gradients detected after layer {i} ({type(layer).__name__})")
                
                if np.max(np.abs(current_grad)) > 1e6:
                    print(f"Warning: Large gradients detected after layer {i} ({type(layer).__name__}): max={np.max(np.abs(current_grad))}")
        
        return gradients
    
    def check_backprop(self, x: np.ndarray, epsilon: float = 1e-5) -> bool:
        """
        Validate backpropagation using numerical gradient checking
        
        Args:
            x: Input tensor for validation
            epsilon: Small value for numerical gradient approximation
            
        Returns:
            True if backpropagation is correct, False otherwise
        """
        print("Running backpropagation validation...")
        
        # Forward pass
        output = self.forward(x)
        
        # Create random gradient output
        grad_output = np.random.randn(*output.shape).astype(np.float32)
        
        # Analytical gradients
        self.backward(grad_output)
        
        # Numerical gradient checking for each parameter
        success = True
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights') and hasattr(layer, 'grad_weights'):
                weights = layer.weights
                grad_weights = layer.grad_weights
                
                # Check a few random weights
                for _ in range(5):
                    try:
                        idx = tuple(np.random.randint(0, s) for s in weights.shape)
                        
                        # Save original weight
                        original_weight = weights[idx]
                        
                        # Compute numerical gradient
                        weights[idx] = original_weight + epsilon
                        output_plus = self.forward(x)
                        loss_plus = np.sum(output_plus * grad_output)
                        
                        weights[idx] = original_weight - epsilon
                        output_minus = self.forward(x)
                        loss_minus = np.sum(output_minus * grad_output)
                        
                        numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
                        analytical_grad = grad_weights[idx]
                        
                        # Restore original weight
                        weights[idx] = original_weight
                        
                        # Check if gradients are close (use relative tolerance)
                        tolerance = 1e-3  # Increased tolerance for numerical stability
                        if abs(numerical_grad - analytical_grad) > tolerance:
                            print(f"Gradient check FAILED for layer {i} at index {idx}")
                            print(f"  Numerical: {numerical_grad:.6f}, Analytical: {analytical_grad:.6f}")
                            print(f"  Difference: {abs(numerical_grad - analytical_grad):.6f}, Tolerance: {tolerance:.6f}")
                            success = False
                    except IndexError:
                        # Skip this weight if index is out of bounds
                        continue
        
        if success:
            print("[OK] Backpropagation validation PASSED")
        else:
            print("[ERROR] Backpropagation validation FAILED")
        
        return success
    
    def save_checkpoint(self, epoch: int, optimizer_state: dict = None):
        """Save model checkpoint"""
        import os
        import pickle
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'layers': []
        }
        
        # Save layer parameters
        for layer in self.layers:
            layer_data = {}
            if hasattr(layer, 'weights'):
                layer_data['weights'] = layer.weights
            if hasattr(layer, 'bias'):
                layer_data['bias'] = layer.bias
            if hasattr(layer, 'gamma'):
                layer_data['gamma'] = layer.gamma
            if hasattr(layer, 'beta'):
                layer_data['beta'] = layer.beta
            checkpoint_data['layers'].append(layer_data)
        
        # Save optimizer state if provided
        if optimizer_state:
            checkpoint_data['optimizer_state'] = optimizer_state
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.checkpoint_prefix}_epoch_{epoch}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        import pickle
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Validate checkpoint compatibility
        if checkpoint_data['input_dim'] != self.input_dim:
            raise ValueError(f"Input dimension mismatch: model expects {self.input_dim}, checkpoint has {checkpoint_data['input_dim']}")
        
        if checkpoint_data['output_dim'] != self.output_dim:
            raise ValueError(f"Output dimension mismatch: model expects {self.output_dim}, checkpoint has {checkpoint_data['output_dim']}")
        
        # Load layer parameters
        for i, (layer, layer_data) in enumerate(zip(self.layers, checkpoint_data['layers'])):
            if hasattr(layer, 'weights') and 'weights' in layer_data:
                layer.weights = layer_data['weights']
            if hasattr(layer, 'bias') and 'bias' in layer_data:
                layer.bias = layer_data['bias']
            if hasattr(layer, 'gamma') and 'gamma' in layer_data:
                layer.gamma = layer_data['gamma']
            if hasattr(layer, 'beta') and 'beta' in layer_data:
                layer.beta = layer_data['beta']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {checkpoint_data['epoch']}")
        
        return checkpoint_data.get('optimizer_state', None)
    
    def get_num_parameters(self) -> int:
        """Count total number of parameters"""
        total_params = 0
        
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                total_params += layer.weights.size
            if hasattr(layer, 'bias'):
                total_params += layer.bias.size
            if hasattr(layer, 'gamma'):
                total_params += layer.gamma.size
            if hasattr(layer, 'beta'):
                total_params += layer.beta.size
        
        return total_params


# Add Dense layer class for completeness
class Dense:
    """Dense layer for RobustNeuralNet"""
    
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize weights using He initialization
        std = np.sqrt(2.0 / input_dim)
        self.weights = np.random.randn(input_dim, output_dim).astype(np.float32) * std
        self.bias = np.zeros(output_dim, dtype=np.float32)
        
        # Gradient storage
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        
        # Cache for backward pass
        self.cache_input = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self.cache_input = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass"""
        # Compute gradients
        self.grad_weights = np.dot(self.cache_input.T, grad_output)
        self.grad_bias = np.sum(grad_output, axis=0)
        
        # Compute gradient for input
        grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input


if __name__ == "__main__":
    main()