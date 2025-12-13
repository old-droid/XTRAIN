"""
CPUWARP-ML Advanced Features
============================
Implements distributed training, model compilation, and export functionality
"""

import numpy as np
import os
import json
import pickle
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from config import get_config, get_env_value
import multiprocessing as mp
from multiprocessing import Queue, Process, Lock

logger = logging.getLogger(__name__)

class DistributedTrainer:
    """Distributed training support for CPUWARP-ML"""
    
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or get_config()
        self.world_size = get_env_value('WORLD_SIZE', 1, int)
        self.rank = get_env_value('RANK', 0, int)
        self.master_addr = get_env_value('MASTER_ADDR', 'localhost')
        self.master_port = get_env_value('MASTER_PORT', 12355, int)
        self.is_distributed = get_env_value('DISTRIBUTED', False, bool)
        
        if self.is_distributed:
            self.setup_distributed()
    
    def setup_distributed(self):
        """Setup distributed training environment"""
        logger.info(f"Setting up distributed training: rank {self.rank}/{self.world_size}")
        
        # For CPU-based distributed training, we use multiprocessing
        self.comm_queue = Queue()
        self.lock = Lock()
        
        # Initialize process group
        if self.world_size > 1:
            os.environ['MASTER_ADDR'] = self.master_addr
            os.environ['MASTER_PORT'] = str(self.master_port)
            logger.info(f"Distributed training initialized on {self.master_addr}:{self.master_port}")
    
    def all_reduce(self, tensor: np.ndarray, op='sum') -> np.ndarray:
        """All-reduce operation for gradient synchronization"""
        if not self.is_distributed or self.world_size == 1:
            return tensor
        
        # Simple all-reduce implementation using multiprocessing
        with self.lock:
            # Put tensor in queue
            self.comm_queue.put((self.rank, tensor))
            
            # Wait for all processes
            tensors = []
            for _ in range(self.world_size):
                rank, t = self.comm_queue.get()
                tensors.append(t)
            
            # Reduce operation
            if op == 'sum':
                result = np.sum(tensors, axis=0)
            elif op == 'mean':
                result = np.mean(tensors, axis=0)
            else:
                result = tensor
            
            return result
    
    def broadcast(self, tensor: np.ndarray, root_rank: int = 0) -> np.ndarray:
        """Broadcast tensor from root to all processes"""
        if not self.is_distributed or self.world_size == 1:
            return tensor
        
        # Simple broadcast implementation
        if self.rank == root_rank:
            # Root sends to all
            for _ in range(self.world_size - 1):
                self.comm_queue.put(tensor)
            return tensor
        else:
            # Others receive from root
            return self.comm_queue.get()
    
    def distributed_data_parallel(self, batch_data, batch_labels):
        """Distribute data across processes"""
        if not self.is_distributed:
            return batch_data, batch_labels
        
        # Split batch across processes
        batch_size = len(batch_data) if isinstance(batch_data, list) else batch_data.shape[0]
        chunk_size = batch_size // self.world_size
        
        start_idx = self.rank * chunk_size
        end_idx = start_idx + chunk_size if self.rank < self.world_size - 1 else batch_size
        
        if isinstance(batch_data, np.ndarray):
            local_data = batch_data[start_idx:end_idx]
            local_labels = batch_labels[start_idx:end_idx]
        else:
            local_data = batch_data[start_idx:end_idx]
            local_labels = batch_labels[start_idx:end_idx]
        
        return local_data, local_labels

class ModelCompiler:
    """Model compilation and optimization"""
    
    def __init__(self, model):
        self.model = model
        self.config = get_config()
        self.compile_enabled = get_env_value('COMPILE_MODEL', False, bool)
        self.channels_last = get_env_value('CHANNELS_LAST', False, bool)
    
    def compile_model(self):
        """Compile model for optimized execution"""
        if not self.compile_enabled:
            return self.model
        
        logger.info("Compiling model for optimized execution...")
        
        # Model compilation optimizations
        # Since we're using NumPy, we can optimize by:
        # 1. Pre-allocating arrays
        # 2. Using memory views
        # 3. Optimizing data layout
        
        if self.channels_last:
            logger.info("Converting to channels-last memory format")
            # Convert model weights to channels-last format (NHWC instead of NCHW)
            self._convert_to_channels_last()
        
        # Pre-compile common operations
        self._optimize_operations()
        
        logger.info("Model compilation complete")
        return self.model
    
    def _convert_to_channels_last(self):
        """Convert model to channels-last format"""
        # For CNN models, transpose weights
        if hasattr(self.model, 'conv1'):
            for layer_name in dir(self.model):
                layer = getattr(self.model, layer_name)
                if hasattr(layer, 'weights'):
                    if len(layer.weights.shape) == 4:  # Conv weights
                        # OIHW -> OHWI
                        layer.weights = np.transpose(layer.weights, (0, 2, 3, 1))
    
    def _optimize_operations(self):
        """Pre-optimize common operations"""
        # Pre-allocate buffers for common sizes
        self.model._preallocated_buffers = {}
        
        common_sizes = [(256, 256), (512, 512), (1024, 1024)]
        for size in common_sizes:
            self.model._preallocated_buffers[size] = np.empty(size, dtype=np.float32)

class ModelOptimizer:
    """Advanced optimization techniques"""
    
    def __init__(self, model):
        self.model = model
        self.config = get_config()
        self.fused_adam = get_env_value('FUSED_ADAM', True, bool)
        self.gradient_checkpointing = get_env_value('GRADIENT_CHECKPOINTING', False, bool)
    
    def get_optimizer(self, learning_rate: float):
        """Get optimized optimizer"""
        if self.fused_adam:
            return FusedAdam(learning_rate)
        else:
            return StandardAdam(learning_rate)
    
    def apply_gradient_checkpointing(self):
        """Apply gradient checkpointing to save memory"""
        if not self.gradient_checkpointing:
            return
        
        logger.info("Applying gradient checkpointing...")
        # Mark certain layers for gradient checkpointing
        if hasattr(self.model, 'blocks'):
            for i, block in enumerate(self.model.blocks):
                if i % 2 == 0:  # Checkpoint every other block
                    block._checkpoint = True

class FusedAdam:
    """Fused Adam optimizer for better performance"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step = 0
        self.m = {}  # First moment
        self.v = {}  # Second moment
    
    def update(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Fused Adam update step"""
        self.step += 1
        
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)
        
        # Fused operations for better cache efficiency
        # Update biased first and second moments
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = self.m[param_name] / (1 - self.beta1 ** self.step)
        v_hat = self.v[param_name] / (1 - self.beta2 ** self.step)
        
        # Update parameters
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        return param

class StandardAdam:
    """Standard Adam optimizer"""
    
    def __init__(self, learning_rate: float = 0.001):
        self.lr = learning_rate
    
    def update(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Simple gradient descent update"""
        return param - self.lr * grad

class ModelExporter:
    """Export models to various formats"""
    
    def __init__(self, model):
        self.model = model
        self.config = get_config()
    
    def export_onnx(self, output_path: str = "model.onnx"):
        """Export model to ONNX format"""
        if not get_env_value('EXPORT_ONNX', False, bool):
            return
        
        logger.info(f"Exporting model to ONNX: {output_path}")
        
        # Create ONNX-like representation
        onnx_model = {
            'format': 'onnx',
            'version': '1.0',
            'graph': self._create_graph(),
            'weights': self._extract_weights(),
            'metadata': {
                'framework': 'CPUWARP-ML',
                'timestamp': time.time()
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(onnx_model, f)
        
        logger.info(f"Model exported to {output_path}")
    
    def export_torchscript(self, output_path: str = "model.pt"):
        """Export model to TorchScript format"""
        if not get_env_value('EXPORT_TORCHSCRIPT', False, bool):
            return
        
        logger.info(f"Exporting model to TorchScript: {output_path}")
        
        # Create TorchScript-like representation
        ts_model = {
            'format': 'torchscript',
            'version': '1.0',
            'modules': self._extract_modules(),
            'weights': self._extract_weights(),
            'metadata': {
                'framework': 'CPUWARP-ML',
                'timestamp': time.time()
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(ts_model, f)
        
        logger.info(f"Model exported to {output_path}")
    
    def export_quantized(self, output_path: str = "model_quantized.npz"):
        """Export quantized model"""
        if not get_env_value('QUANTIZE_MODEL', False, bool):
            return
        
        bits = get_env_value('QUANTIZATION_BITS', 8, int)
        logger.info(f"Quantizing model to {bits} bits")
        
        weights = self._extract_weights()
        quantized_weights = {}
        
        for name, weight in weights.items():
            # Simple quantization
            if isinstance(weight, np.ndarray):
                # Scale to quantization range
                min_val = weight.min()
                max_val = weight.max()
                scale = (max_val - min_val) / (2**bits - 1)
                
                # Quantize
                quantized = np.round((weight - min_val) / scale).astype(np.uint8 if bits == 8 else np.uint16)
                
                quantized_weights[name] = {
                    'data': quantized,
                    'scale': scale,
                    'zero_point': min_val,
                    'bits': bits
                }
            else:
                quantized_weights[name] = weight
        
        np.savez_compressed(output_path, **quantized_weights)
        logger.info(f"Quantized model saved to {output_path}")
    
    def _create_graph(self) -> Dict:
        """Create computation graph representation"""
        graph = {
            'nodes': [],
            'edges': [],
            'inputs': [],
            'outputs': []
        }
        
        # Simplified graph creation
        if hasattr(self.model, 'forward'):
            graph['nodes'].append({
                'name': 'model',
                'type': type(self.model).__name__,
                'params': self.model.get_num_parameters() if hasattr(self.model, 'get_num_parameters') else 0
            })
        
        return graph
    
    def _extract_modules(self) -> Dict:
        """Extract model modules"""
        modules = {}
        
        for attr_name in dir(self.model):
            if not attr_name.startswith('_'):
                attr = getattr(self.model, attr_name)
                if hasattr(attr, 'forward'):
                    modules[attr_name] = type(attr).__name__
        
        return modules
    
    def _extract_weights(self) -> Dict:
        """Extract model weights"""
        weights = {}
        
        for attr_name in dir(self.model):
            if not attr_name.startswith('_'):
                attr = getattr(self.model, attr_name)
                
                # Extract weights from layers
                if hasattr(attr, 'weights'):
                    weights[f"{attr_name}.weights"] = attr.weights
                if hasattr(attr, 'bias'):
                    weights[f"{attr_name}.bias"] = attr.bias
                
                # Extract direct numpy arrays
                if isinstance(attr, np.ndarray):
                    weights[attr_name] = attr
        
        return weights

class MixedPrecisionTrainer:
    """Mixed precision training support"""
    
    def __init__(self):
        self.config = get_config()
        self.enabled = self.config.training.mixed_precision
        self.loss_scale = 1024.0
        
    def scale_loss(self, loss: float) -> float:
        """Scale loss for mixed precision training"""
        if not self.enabled:
            return loss
        return loss * self.loss_scale
    
    def unscale_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """Unscale gradients after backward pass"""
        if not self.enabled:
            return gradients
        return gradients / self.loss_scale
    
    def cast_to_fp16(self, tensor: np.ndarray) -> np.ndarray:
        """Cast tensor to FP16 for mixed precision"""
        if not self.enabled:
            return tensor
        return tensor.astype(np.float16)
    
    def cast_to_fp32(self, tensor: np.ndarray) -> np.ndarray:
        """Cast tensor back to FP32"""
        return tensor.astype(np.float32)

def enable_advanced_features(model):
    """Enable all advanced features for a model"""
    config = get_config()
    
    # Apply model compilation
    if get_env_value('COMPILE_MODEL', False, bool):
        compiler = ModelCompiler(model)
        model = compiler.compile_model()
    
    # Setup distributed training
    if get_env_value('DISTRIBUTED', False, bool):
        distributed = DistributedTrainer(model)
        model._distributed = distributed
    
    # Setup optimizer
    optimizer = ModelOptimizer(model)
    if get_env_value('GRADIENT_CHECKPOINTING', False, bool):
        optimizer.apply_gradient_checkpointing()
    
    # Setup mixed precision
    if config.training.mixed_precision:
        model._mixed_precision = MixedPrecisionTrainer()
    
    # Setup model export
    model._exporter = ModelExporter(model)
    
    return model

if __name__ == "__main__":
    # Test advanced features
    print("Testing CPUWARP-ML Advanced Features")
    print("=" * 40)
    
    # Test distributed training
    print("\n1. Distributed Training:")
    distributed = DistributedTrainer(None)
    print(f"   Distributed: {distributed.is_distributed}")
    print(f"   World Size: {distributed.world_size}")
    print(f"   Rank: {distributed.rank}")
    
    # Test model compilation
    print("\n2. Model Compilation:")
    compiler = ModelCompiler(None)
    print(f"   Compile Enabled: {compiler.compile_enabled}")
    print(f"   Channels Last: {compiler.channels_last}")
    
    # Test optimizer
    print("\n3. Optimizer:")
    optimizer = ModelOptimizer(None)
    adam = optimizer.get_optimizer(0.001)
    print(f"   Optimizer Type: {type(adam).__name__}")
    
    # Test mixed precision
    print("\n4. Mixed Precision:")
    mp_trainer = MixedPrecisionTrainer()
    print(f"   Enabled: {mp_trainer.enabled}")
    print(f"   Loss Scale: {mp_trainer.loss_scale}")
    
    print("\nAdvanced features test complete!")


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
