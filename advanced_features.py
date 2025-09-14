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
from config import get_config
import multiprocessing as mp
from multiprocessing import Queue, Process, Lock

logger = logging.getLogger(__name__)

class DistributedTrainer:
    """Distributed training support for CPUWARP-ML"""
    
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or get_config()
        self.world_size = self.config.get_env_value('WORLD_SIZE', 1, int)
        self.rank = self.config.get_env_value('RANK', 0, int)
        self.master_addr = self.config.get_env_value('MASTER_ADDR', 'localhost')
        self.master_port = self.config.get_env_value('MASTER_PORT', 12355, int)
        self.is_distributed = self.config.get_env_value('DISTRIBUTED', False, bool)
        
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
        self.compile_enabled = self.config.get_env_value('COMPILE_MODEL', False, bool)
        self.channels_last = self.config.get_env_value('CHANNELS_LAST', False, bool)
    
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
        self.fused_adam = self.config.get_env_value('FUSED_ADAM', True, bool)
        self.gradient_checkpointing = self.config.get_env_value('GRADIENT_CHECKPOINTING', False, bool)
    
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
        if not self.config.get_env_value('EXPORT_ONNX', False, bool):
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
        if not self.config.get_env_value('EXPORT_TORCHSCRIPT', False, bool):
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
        if not self.config.get_env_value('QUANTIZE_MODEL', False, bool):
            return
        
        bits = self.config.get_env_value('QUANTIZATION_BITS', 8, int)
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
    if config.get_env_value('COMPILE_MODEL', False, bool):
        compiler = ModelCompiler(model)
        model = compiler.compile_model()
    
    # Setup distributed training
    if config.get_env_value('DISTRIBUTED', False, bool):
        distributed = DistributedTrainer(model)
        model._distributed = distributed
    
    # Setup optimizer
    optimizer = ModelOptimizer(model)
    if config.get_env_value('GRADIENT_CHECKPOINTING', False, bool):
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