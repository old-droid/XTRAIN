#!/usr/bin/env python3
"""
Raptor Metrics Implementation
Comprehensive metrics for LM loss, RL score, efficiency, and performance comparison
"""

import numpy as np
import time
import psutil
import os
from typing import Dict, List, Tuple, Any, Optional
import sys

# Add current directory to path
sys.path.append(".")

from Raptor_Xtrain import Decoder_LLM as XtrainDecoderLLM


class RaptorMetrics:
    """Comprehensive metrics tracker for Raptor models."""
    
    def __init__(self):
        self.metrics_history = []
        self.start_time = None
        self.start_memory = None
        
    def start_tracking(self):
        """Start tracking metrics."""
        self.start_time = time.time()
        process = psutil.Process(os.getpid())
        self.start_memory = process.memory_info().rss
        
    def end_tracking(self, model_name: str = "unknown"):
        """End tracking and record metrics."""
        if self.start_time is None:
            return None
            
        end_time = time.time()
        process = psutil.Process(os.getpid())
        end_memory = process.memory_info().rss
        
        metrics = {
            "model": model_name,
            "time": end_time - self.start_time,
            "memory_used": (end_memory - self.start_memory) / 1024 / 1024,  # MB
            "timestamp": time.time()
        }
        
        self.metrics_history.append(metrics)
        return metrics
        
    def get_latest_metrics(self) -> Optional[Dict]:
        """Get the latest recorded metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
        
    def get_all_metrics(self) -> List[Dict]:
        """Get all recorded metrics."""
        return self.metrics_history
        
    def calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score (lower is better)."""
        if not self.metrics_history:
            return 0.0
            
        total_time = sum(m["time"] for m in self.metrics_history)
        total_memory = sum(m["memory_used"] for m in self.metrics_history)
        
        # Efficiency = time * memory (lower is better)
        efficiency_score = total_time * total_memory
        return float(efficiency_score)
        
    def reset(self):
        """Reset all metrics."""
        self.metrics_history = []
        self.start_time = None
        self.start_memory = None


def calculate_lm_loss(logits: np.ndarray, labels: np.ndarray, ignore_index: int = -100) -> float:
    """
    Calculate language modeling loss (cross-entropy).
    
    Args:
        logits: Model output logits (B, L, vocab_size)
        labels: Target labels (B, L)
        ignore_index: Index to ignore in loss calculation
        
    Returns:
        LM loss value
    """
    import cpuwarp_ml
    
    B, L, vocab_size = logits.shape
    
    # Create one-hot labels
    one_hot_labels = np.zeros((B, L, vocab_size), dtype=np.float32)
    mask = labels != ignore_index
    
    for i in range(B):
        for j in range(L):
            if mask[i, j]:
                one_hot_labels[i, j, labels[i, j]] = 1.0
    
    # Calculate softmax and cross-entropy
    probs = cpuwarp_ml.softmax(logits, axis=-1)
    
    # Add small epsilon to avoid log(0)
    log_probs = np.log(probs + 1e-9)
    
    # Calculate loss only for non-ignored tokens
    loss_per_token = -np.sum(one_hot_labels * log_probs, axis=-1)
    total_loss = np.sum(loss_per_token * mask) / (np.sum(mask) + 1e-6)
    
    return float(total_loss)


def calculate_lb_loss(probs: np.ndarray, n_experts: int) -> float:
    """
    Calculate load balancing loss for MoE.
    
    Args:
        probs: Router probabilities (n_tokens, n_experts)
        n_experts: Number of experts
        
    Returns:
        Load balancing loss value
    """
    # Calculate importance per expert
    importance = probs.sum(axis=0)
    imp = importance / probs.shape[0]
    
    # Load balancing loss
    lb_loss = (n_experts * (imp * imp).sum()).astype(np.float32)
    
    return float(lb_loss)


def calculate_rl_loss(log_probs: List[np.ndarray], reward: float) -> float:
    """
    Calculate reinforcement learning loss.
    
    Args:
        log_probs: List of log probabilities from each layer (each: B, L)
        reward: Reward signal (scalar)
        
    Returns:
        RL loss value
    """
    rl_loss = 0.0
    
    for lp in log_probs:
        # Flatten and calculate RL loss
        flat_lp = lp.reshape(-1)
        rl_loss += -np.mean(reward * flat_lp)
    
    return float(rl_loss)


def calculate_rl_score(reward: float, rl_loss: float, epsilon: float = 1e-6) -> float:
    """
    Calculate RL score (higher is better).
    
    Args:
        reward: Reward value
        rl_loss: RL loss value
        epsilon: Small value to avoid division by zero
        
    Returns:
        RL score
    """
    # RL score = reward / (1 + |rl_loss|)
    # Higher reward and lower loss give better score
    denominator = 1.0 + abs(rl_loss) + epsilon
    rl_score = reward / denominator
    
    return float(rl_score)


def calculate_perplexity(lm_loss: float) -> float:
    """
    Calculate perplexity from LM loss.
    
    Args:
        lm_loss: Language modeling loss
        
    Returns:
        Perplexity score
    """
    return float(np.exp(lm_loss))


def calculate_efficiency_metrics(
    computation_time: float,
    memory_usage: float,
    lm_loss: float,
    rl_score: float
) -> Dict[str, float]:
    """
    Calculate comprehensive efficiency metrics.
    
    Args:
        computation_time: Time taken for computation (seconds)
        memory_usage: Memory used (MB)
        lm_loss: Language modeling loss
        rl_score: RL score
        
    Returns:
        Dictionary of efficiency metrics
    """
    # Time-memory efficiency (lower is better)
    time_memory_efficiency = computation_time * memory_usage
    
    # Loss-efficiency tradeoff (lower is better)
    loss_efficiency = lm_loss * computation_time
    
    # Overall efficiency score (higher is better)
    # Combines speed, memory, and quality
    overall_efficiency = (
        (1.0 / (time_memory_efficiency + 1e-6)) * 
        (1.0 / (lm_loss + 1e-6)) * 
        (rl_score + 1.0)
    )
    
    return {
        "time_memory_efficiency": float(time_memory_efficiency),
        "loss_efficiency": float(loss_efficiency),
        "overall_efficiency": float(overall_efficiency),
        "time_per_loss_unit": float(computation_time / (lm_loss + 1e-6)),
        "memory_per_loss_unit": float(memory_usage / (lm_loss + 1e-6))
    }


def benchmark_model_performance(
    model: XtrainDecoderLLM,
    input_ids: np.ndarray,
    labels: np.ndarray,
    n_iterations: int = 10,
    warmup_iterations: int = 3
) -> Dict[str, Any]:
    """
    Comprehensive benchmark for model performance.
    
    Args:
        model: Model to benchmark
        input_ids: Input data
        labels: Target labels
        n_iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations
        
    Returns:
        Dictionary of performance metrics
    """
    print(f"Benchmarking model with {n_iterations} iterations...")
    
    # Warmup
    for _ in range(warmup_iterations):
        _ = model.forward(input_ids)
    
    # Track metrics
    metrics_tracker = RaptorMetrics()
    
    # Time forward passes
    forward_times = []
    for i in range(n_iterations):
        start_time = time.time()
        outputs = model.forward(input_ids, return_log_probs=True)
        forward_time = time.time() - start_time
        forward_times.append(forward_time)
        
        if i == 0:  # Only calculate metrics once
            logits, all_log_probs, total_lb_loss = outputs
            
            # Calculate losses
            lm_loss = calculate_lm_loss(logits, labels)
            
            # For RL, we need a reward. Use negative LM loss as reward
            reward = -lm_loss
            rl_loss = calculate_rl_loss(all_log_probs, reward)
            rl_score = calculate_rl_score(reward, rl_loss)
            
            # Calculate other metrics
            perplexity = calculate_perplexity(lm_loss)
            
    # Calculate efficiency metrics
    avg_forward_time = np.mean(forward_times)
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    
    efficiency_metrics = calculate_efficiency_metrics(
        avg_forward_time,
        memory_usage,
        lm_loss,
        rl_score
    )
    
    # Compile all results
    results = {
        "performance": {
            "avg_forward_time": float(avg_forward_time),
            "forward_times": [float(t) for t in forward_times],
            "std_forward_time": float(np.std(forward_times)),
            "min_forward_time": float(np.min(forward_times)),
            "max_forward_time": float(np.max(forward_times)),
            "memory_usage": float(memory_usage)
        },
        "losses": {
            "lm_loss": float(lm_loss),
            "lb_loss": float(total_lb_loss),
            "rl_loss": float(rl_loss),
            "perplexity": float(perplexity)
        },
        "rl_metrics": {
            "reward": float(reward),
            "rl_score": float(rl_score)
        },
        "efficiency": efficiency_metrics,
        "input_shape": input_ids.shape,
        "model_config": {
            "vocab_size": model.vocab_size,
            "d_model": model.d_model,
            "n_layers": model.n_layers,
            "n_experts": len(model.layers[0].moe.experts) if model.layers else 0
        }
    }
    
    return results


def compare_models_pytorch_vs_xtrain(
    pytorch_model: Any = None,
    xtrain_model: XtrainDecoderLLM = None,
    input_ids: np.ndarray = None,
    labels: np.ndarray = None,
    n_iterations: int = 5
) -> Dict[str, Any]:
    """
    Compare PyTorch and Xtrain model performance.
    
    Args:
        pytorch_model: PyTorch model (optional)
        xtrain_model: Xtrain model (optional)
        input_ids: Input data
        labels: Target labels
        n_iterations: Number of iterations
        
    Returns:
        Comparison results
    """
    comparison = {}
    
    # Test PyTorch if available
    if pytorch_model is not None:
        try:
            import torch
            import torch.nn.functional as F
            
            print("Benchmarking PyTorch model...")
            
            # Convert to tensor
            input_tensor = torch.from_numpy(input_ids).long()
            labels_tensor = torch.from_numpy(labels).long()
            
            # Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pytorch_model = pytorch_model.to(device)
            input_tensor = input_tensor.to(device)
            labels_tensor = labels_tensor.to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(2):
                    _ = pytorch_model(input_tensor)
            
            # Benchmark
            forward_times = []
            for _ in range(n_iterations):
                start_time = time.time()
                with torch.no_grad():
                    outputs = pytorch_model(input_tensor, return_log_probs=True)
                forward_time = time.time() - start_time
                forward_times.append(forward_time)
                
                if len(forward_times) == 1:  # First iteration
                    logits = outputs["logits"]
                    all_log_probs = outputs[1]  # log_probs
                    lb_loss = outputs[2]  # lb_loss
                    
                    # Calculate LM loss
                    lm_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels_tensor.view(-1),
                        ignore_index=-100
                    ).item()
                    
                    # Calculate RL metrics
                    reward = -lm_loss
                    rl_loss = 0.0
                    for lp in all_log_probs:
                        flat_lp = lp.view(-1)
                        rl_loss += -(torch.tensor(reward) * flat_lp).mean().item()
                    
                    rl_score = calculate_rl_score(reward, rl_loss)
                    perplexity = calculate_perplexity(lm_loss)
            
            # Memory usage
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            comparison["pytorch"] = {
                "avg_time": float(np.mean(forward_times)),
                "lm_loss": float(lm_loss),
                "lb_loss": float(lb_loss),
                "rl_loss": float(rl_loss),
                "rl_score": float(rl_score),
                "perplexity": float(perplexity),
                "memory_usage": float(memory_usage),
                "device": str(device)
            }
            
        except Exception as e:
            print(f"PyTorch benchmark failed: {e}")
            comparison["pytorch"] = {"error": str(e)}
    
    # Test Xtrain if available
    if xtrain_model is not None:
        try:
            print("Benchmarking Xtrain model...")
            
            # Benchmark Xtrain model
            xtrain_results = benchmark_model_performance(
                xtrain_model, input_ids, labels, n_iterations
            )
            
            comparison["xtrain"] = {
                "avg_time": xtrain_results["performance"]["avg_forward_time"],
                "lm_loss": xtrain_results["losses"]["lm_loss"],
                "lb_loss": xtrain_results["losses"]["lb_loss"],
                "rl_loss": xtrain_results["losses"]["rl_loss"],
                "rl_score": xtrain_results["rl_metrics"]["rl_score"],
                "perplexity": xtrain_results["losses"]["perplexity"],
                "memory_usage": xtrain_results["performance"]["memory_usage"],
                "device": "CPU"
            }
            
        except Exception as e:
            print(f"Xtrain benchmark failed: {e}")
            comparison["xtrain"] = {"error": str(e)}
    
    # Calculate comparison metrics
    if "pytorch" in comparison and "xtrain" in comparison:
        if "error" not in comparison["pytorch"] and "error" not in comparison["xtrain"]:
            pytorch = comparison["pytorch"]
            xtrain = comparison["xtrain"]
            
            # Speed comparison
            speedup = pytorch["avg_time"] / xtrain["avg_time"]
            comparison["speedup"] = speedup
            comparison["faster"] = "xtrain" if speedup > 1 else "pytorch"
            
            # Memory comparison
            memory_ratio = pytorch["memory_usage"] / xtrain["memory_usage"]
            comparison["memory_ratio"] = memory_ratio
            comparison["memory_efficient"] = "xtrain" if memory_ratio > 1 else "pytorch"
            
            # Quality comparison (lower LM loss is better)
            lm_loss_ratio = xtrain["lm_loss"] / pytorch["lm_loss"]
            comparison["lm_loss_ratio"] = lm_loss_ratio
            comparison["better_lm"] = "xtrain" if lm_loss_ratio < 1 else "pytorch"
            
            # RL score comparison (higher is better)
            rl_score_ratio = xtrain["rl_score"] / pytorch["rl_score"]
            comparison["rl_score_ratio"] = rl_score_ratio
            comparison["better_rl"] = "xtrain" if rl_score_ratio > 1 else "pytorch"
            
            # Overall efficiency
            pytorch_efficiency = pytorch["avg_time"] * pytorch["memory_usage"] / (pytorch["lm_loss"] + 1e-6)
            xtrain_efficiency = xtrain["avg_time"] * xtrain["memory_usage"] / (xtrain["lm_loss"] + 1e-6)
            
            efficiency_ratio = xtrain_efficiency / pytorch_efficiency
            comparison["efficiency_ratio"] = efficiency_ratio
            comparison["more_efficient"] = "xtrain" if efficiency_ratio < 1 else "pytorch"
    
    return comparison


def print_comparison_results(comparison: Dict[str, Any]):
    """Print comparison results in a readable format."""
    print("\n" + "=" * 80)
    print("RAPTOR MODEL COMPARISON: PYTORCH vs XTRAIN")
    print("=" * 80)
    
    if "pytorch" in comparison and "xtrain" in comparison:
        pytorch = comparison.get("pytorch", {})
        xtrain = comparison.get("xtrain", {})
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  PyTorch ({pytorch.get('device', 'N/A')}): {pytorch.get('avg_time', 0):.4f}s avg")
        print(f"  Xtrain  (CPU):                    {xtrain.get('avg_time', 0):.4f}s avg")
        
        if "speedup" in comparison:
            if comparison["speedup"] > 1:
                print(f"  ✅ Xtrain is {comparison['speedup']:.2f}x FASTER")
            else:
                print(f"  ❌ Xtrain is {1/comparison['speedup']:.2f}x SLOWER")
        
        print(f"\n🧠 LANGUAGE MODELING:")
        print(f"  PyTorch LM Loss: {pytorch.get('lm_loss', 0):.4f}")
        print(f"  Xtrain  LM Loss: {xtrain.get('lm_loss', 0):.4f}")
        print(f"  PyTorch Perplexity: {pytorch.get('perplexity', 0):.2f}")
        print(f"  Xtrain  Perplexity: {xtrain.get('perplexity', 0):.2f}")
        
        if "better_lm" in comparison:
            if comparison["better_lm"] == "xtrain":
                print(f"  ✅ Xtrain has better language modeling quality")
            else:
                print(f"  ❌ PyTorch has better language modeling quality")
        
        print(f"\n🤖 REINFORCEMENT LEARNING:")
        print(f"  PyTorch RL Score: {pytorch.get('rl_score', 0):.4f}")
        print(f"  Xtrain  RL Score: {xtrain.get('rl_score', 0):.4f}")
        
        if "better_rl" in comparison:
            if comparison["better_rl"] == "xtrain":
                print(f"  ✅ Xtrain has better RL performance")
            else:
                print(f"  ❌ PyTorch has better RL performance")
        
        print(f"\n💾 MEMORY USAGE:")
        print(f"  PyTorch: {pytorch.get('memory_usage', 0):.2f} MB")
        print(f"  Xtrain:  {xtrain.get('memory_usage', 0):.2f} MB")
        
        if "memory_efficient" in comparison:
            if comparison["memory_efficient"] == "xtrain":
                print(f"  ✅ Xtrain is more memory efficient")
            else:
                print(f"  ❌ PyTorch is more memory efficient")
        
        print(f"\n⚡ OVERALL EFFICIENCY:")
        if "more_efficient" in comparison:
            if comparison["more_efficient"] == "xtrain":
                efficiency_improvement = (1 - comparison["efficiency_ratio"]) * 100
                print(f"  ✅ Xtrain is {efficiency_improvement:.1f}% more efficient overall")
            else:
                efficiency_degradation = (comparison["efficiency_ratio"] - 1) * 100
                print(f"  ❌ Xtrain is {efficiency_degradation:.1f}% less efficient overall")
    
    print("\n" + "=" * 80)


def create_demo_comparison():
    """Create a demo comparison with sample models."""
    print("Creating demo comparison...")
    
    # Create test data
    batch_size = 2
    seq_len = 16
    vocab_size = 100
    
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int32)
    labels = np.zeros_like(input_ids)
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    
    # Create Xtrain model
    xtrain_model = XtrainDecoderLLM(
        vocab_size=vocab_size,
        d_model=64,
        max_seq_len=seq_len,
        n_layers=2,
        d_ff=128,
        n_heads=2,
        top_k_tokens=1,
        n_experts=2,
        k_moe=1
    )
    
    # Try to create PyTorch model
    pytorch_model = None
    try:
        from Raptor_V6 import Decoder_LLM as PyTorchDecoderLLM
        import torch
        
        pytorch_model = PyTorchDecoderLLM(
            vocab_size=vocab_size,
            d_model=64,
            max_seq_len=seq_len,
            n_layers=2,
            d_ff=128,
            n_heads=2,
            top_k_tokens=1,
            n_experts=2,
            k_moe=1
        )
        
    except ImportError:
        print("PyTorch not available for demo comparison")
    
    # Run comparison
    comparison = compare_models_pytorch_vs_xtrain(
        pytorch_model=pytorch_model,
        xtrain_model=xtrain_model,
        input_ids=input_ids,
        labels=labels,
        n_iterations=3
    )
    
    # Print results
    print_comparison_results(comparison)
    
    return comparison


if __name__ == "__main__":
    print("Raptor Metrics and Performance Analysis")
    print("=" * 50)
    
    # Run demo comparison
    demo_results = create_demo_comparison()
    
    print("\nSUMMARY:")
    print("This metrics framework provides comprehensive analysis of:")
    print("  • Language Modeling Loss (LM Loss)")
    print("  • Reinforcement Learning Score (RL Score)")
    print("  • Computational Efficiency")
    print("  • Memory Usage")
    print("  • Overall Performance")
    print("\nUse these metrics to compare PyTorch and Xtrain implementations!")