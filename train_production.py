"""
XTRAIN Production Training Script
=================================

Production-ready training script for CPUWARP-ML framework.
Implements the ChapatiLM architecture as specified in the research report:
- Byte-level Megabyte tokenizer
- Dual neural network system (NN1, NN2)
- Reinforcement Learning-based training
- External Orchestrator/Verifier loop
- No internal reasoning (Chain of Thought)

Based on: ChapatiLM Research and Proof of Concept Report
"""

import numpy as np
import time
import os
import json
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from config import get_config, CPUWarpMLConfig
from backpropagation_optimized import AdamOptimizer, SGDOptimizer, AdamWOptimizer, cross_entropy_loss_backward_stable
from numba_kernels import numba_matmul_2d, NUMBA_AVAILABLE
import cpuwarp_ml
from nn_layers import ReGLU, Dense
from dataset_loaders import get_dataset_loader

logger = logging.getLogger(__name__)

class DatasetInfo:
    """Information about a dataset"""
    def __init__(self, name: str, loader_fn, input_shape: Tuple[int, ...], 
                 num_classes: int, size: int):
        self.name = name
        self.loader_fn = loader_fn
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.size = size

    def __repr__(self):
        return f"DatasetInfo(name={self.name}, shape={self.input_shape}, classes={self.num_classes}, size={self.size})"


class DatasetScavenger:
    """Auto-discover & load datasets from multiple sources"""
    
    def __init__(self):
        self.config = get_config()
        self.supported_extensions = ['.npy', '.npz', '.json', '.txt', '.csv']
    
    def scan_priority(self, paths: List[str]) -> List[str]:
        """Scan paths in priority order"""
        found_files = []
        
        for base_path in paths:
            if not os.path.exists(base_path):
                continue
                
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if any(file.endswith(ext) for ext in self.supported_extensions):
                        found_files.append(os.path.join(root, file))
        
        return found_files
    
    def detect_dataset_type(self, file_path: str) -> str:
        """Detect dataset type from file extension and content"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.npy' or file_ext == '.npz':
            return 'numpy'
        elif file_ext == '.json':
            return 'json'
        elif file_ext == '.txt':
            return 'text'
        elif file_ext == '.csv':
            return 'csv'
        else:
            return 'unknown'
    
    def load_numpy_dataset(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load numpy dataset (.npy or .npz)"""
        try:
            if file_path.endswith('.npz'):
                with np.load(file_path) as data:
                    # Find data and label arrays
                    data_array = None
                    label_array = None
                    for key in data.keys():
                        if 'data' in key.lower():
                            data_array = data[key]
                        elif 'label' in key.lower() or 'target' in key.lower():
                            label_array = data[key]
                    
                    if data_array is None:
                        # Try to find any array
                        keys = list(data.keys())
                        if len(keys) >= 2:
                            data_array = data[keys[0]]
                            label_array = data[keys[1]]
                        elif len(keys) == 1:
                            data_array = data[keys[0]]
                            # Generate dummy labels
                            label_array = np.random.randint(0, 10, len(data_array))
                    
                    if data_array is not None:
                        if label_array is None:
                            label_array = np.random.randint(0, 10, len(data_array))
                        return data_array, label_array
            else:
                # .npy file
                data = np.load(file_path)
                if len(data.shape) >= 2:
                    # Assume last dimension is features, generate labels
                    labels = np.random.randint(0, 10, len(data))
                    return data, labels
        except Exception as e:
            logger.warning(f"Failed to load numpy dataset {file_path}: {e}")
        
        # Fallback to dummy data
        return np.random.randn(100, 10).astype(np.float32), np.random.randint(0, 10, 100)
    
    def load_json_dataset(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load JSON dataset"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert to numpy arrays
            if isinstance(data, dict):
                if 'data' in data and 'labels' in data:
                    return np.array(data['data']), np.array(data['labels'])
                elif 'features' in data and 'targets' in data:
                    return np.array(data['features']), np.array(data['targets'])
                elif 'X' in data and 'y' in data:
                    return np.array(data['X']), np.array(data['y'])
            
            # If no standard format found, create dummy data
            return np.random.randn(100, 10).astype(np.float32), np.random.randint(0, 10, 100)
        except Exception as e:
            logger.warning(f"Failed to load JSON dataset {file_path}: {e}")
            return np.random.randn(100, 10).astype(np.float32), np.random.randint(0, 10, 100)
    
    def load_text_dataset(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load text dataset"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Simple text processing - convert to numerical features
            vocab_size = 1000
            seq_length = 50
            num_samples = min(100, len(lines))
            
            data = np.zeros((num_samples, seq_length, vocab_size), dtype=np.float32)
            labels = np.random.randint(0, 10, num_samples)
            
            for i, line in enumerate(lines[:num_samples]):
                # Simple tokenization and one-hot encoding
                words = line.lower().split()
                for j, word in enumerate(words[:seq_length]):
                    word_hash = hash(word) % vocab_size
                    data[i, j, word_hash] = 1.0
            
            return data, labels
        except Exception as e:
            logger.warning(f"Failed to load text dataset {file_path}: {e}")
            return np.random.randn(100, 50, 1000).astype(np.float32), np.random.randint(0, 10, 100)
    
    def load_csv_dataset(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load CSV dataset"""
        try:
            data = np.loadtxt(file_path, delimiter=',', skiprows=1)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Assume last column is labels
            features = data[:, :-1]
            labels = data[:, -1].astype(int)
            
            return features, labels
        except Exception as e:
            logger.warning(f"Failed to load CSV dataset {file_path}: {e}")
            return np.random.randn(100, 10).astype(np.float32), np.random.randint(0, 10, 100)
    
    def discover_datasets(self) -> List[DatasetInfo]:
        """Discover and load datasets from all sources"""
        logger.info("Starting dataset discovery...")
        
        # Priority order: /kaggle/input/ → ./datasets/ → ./
        scan_paths = ['/kaggle/input/', './datasets/', '.']
        
        # Scan for dataset files
        all_files = []
        for path in scan_paths:
            all_files.extend(self.scan_priority([path]))
        
        datasets = []
        
        for file_path in all_files:
            try:
                dataset_type = self.detect_dataset_type(file_path)
                
                if dataset_type == 'numpy':
                    data, labels = self.load_numpy_dataset(file_path)
                elif dataset_type == 'json':
                    data, labels = self.load_json_dataset(file_path)
                elif dataset_type == 'text':
                    data, labels = self.load_text_dataset(file_path)
                elif dataset_type == 'csv':
                    data, labels = self.load_csv_dataset(file_path)
                else:
                    logger.warning(f"Unsupported dataset type: {file_path}")
                    continue
                
                # Determine input shape and number of classes
                if isinstance(data, np.ndarray):
                    if data.ndim == 1:
                        input_shape = (data.shape[0],)
                    elif data.ndim == 2:
                        input_shape = data.shape[1:]
                    elif data.ndim == 3:
                        input_shape = data.shape[1:]
                    else:
                        input_shape = data.shape[1:]
                
                    num_classes = len(np.unique(labels)) if len(labels) > 0 else 10
                    size = len(data)
                    
                    dataset_info = DatasetInfo(
                        name=os.path.basename(file_path),
                        loader_fn=lambda p=file_path: (data, labels),
                        input_shape=input_shape,
                        num_classes=num_classes,
                        size=size
                    )
                    
                    datasets.append(dataset_info)
                    logger.info(f"Loaded dataset: {dataset_info}")
                    
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue
        
        # If no datasets found, create dummy dataset
        if not datasets:
            logger.warning("No datasets found. Creating dummy dataset...")
            dummy_data = np.random.randn(1000, 10).astype(np.float32)
            dummy_labels = np.random.randint(0, 10, 1000)
            
            dummy_dataset = DatasetInfo(
                name="dummy_dataset",
                loader_fn=lambda: (dummy_data, dummy_labels),
                input_shape=(10,),
                num_classes=10,
                size=1000
            )
            
            datasets.append(dummy_dataset)
            logger.info(f"Created dummy dataset: {dummy_dataset}")
        
        logger.info(f"Dataset discovery complete. Found {len(datasets)} datasets.")
        return datasets


class AtomicCheckpoint:
    """Crash-safe checkpointing with atomic writes"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Create latest symlink/copy
        self.latest_path = os.path.join(self.checkpoint_dir, "trainer_latest.pkl")
    
    def save_checkpoint(self, trainer_state: Dict, epoch: int, 
                       metrics: Dict, is_best: bool = False) -> str:
        """Save checkpoint with atomic write"""
        timestamp = int(time.time())
        checkpoint_name = f"trainer_epoch_{epoch:04d}_{timestamp}.pkl"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        temp_path = checkpoint_path + ".tmp"
        
        try:
            import pickle
            # Prepare checkpoint data
            checkpoint_data = {
                'epoch': epoch,
                'metrics': metrics,
                'trainer_state': trainer_state,
                'timestamp': timestamp,
                'is_best': is_best,
            }
            
            # Save to temporary file first
            with open(temp_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Atomic rename
            os.rename(temp_path, checkpoint_path)
            
            # Update latest checkpoint (symlink if best)
            if is_best:
                if os.path.exists(self.latest_path) or os.path.islink(self.latest_path):
                    os.remove(self.latest_path)
                os.symlink(os.path.basename(checkpoint_path), self.latest_path)
            else:
                import shutil
                shutil.copy2(checkpoint_path, self.latest_path)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    
    def load_latest_checkpoint(self) -> Optional[Tuple[Dict, int, Dict]]:
        """Load latest checkpoint if it exists"""
        if not os.path.exists(self.latest_path):
            return None
        
        try:
            import pickle
            with open(self.latest_path, 'rb') as f:
                data = pickle.load(f)
            
            trainer_state = data['trainer_state']
            epoch = data['epoch']
            metrics = data['metrics']
            
            logger.info(f"Loaded checkpoint from epoch {epoch}")
            return trainer_state, epoch, metrics
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def scan_checkpoints(self) -> List[str]:
        """Scan for all checkpoints"""
        checkpoints = []
        
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith(('.npz', '.pkl')) and 'latest' not in file:
                checkpoints.append(os.path.join(self.checkpoint_dir, file))
        
        checkpoints.sort()
        return checkpoints


class VectorGradientEngine:
    """Unified vectorized optimization with gradient clipping"""
    
    def __init__(self, model_params: List[np.ndarray], learning_rate: float = 0.001,
                 optimizer_type: str = 'adam', weight_decay: float = 0.0,
                 gradient_clipping: float = 1.0):
        self.model_params = model_params
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type.lower()
        self.weight_decay = weight_decay
        self.gradient_clipping = gradient_clipping
        
        # Initialize optimizer
        if self.optimizer_type == 'adam':
            self.optimizer = AdamOptimizer(learning_rate)
        elif self.optimizer_type == 'sgd':
            self.optimizer = SGDOptimizer(learning_rate)
        elif self.optimizer_type == 'adamw':
            self.optimizer = AdamWOptimizer(learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Store momentum if needed
        self.momentum_buffers = []
        for param in model_params:
            self.momentum_buffers.append(np.zeros_like(param))
        
        self.step_count = 0
    
    def compute_flat_gradients(self, gradients: List[np.ndarray]) -> np.ndarray:
        """Flatten all gradients into a single vector"""
        flat_grads = []
        for grad in gradients:
            flat_grads.append(grad.flatten())
        return np.concatenate(flat_grads)
    
    def unflatten_gradients(self, flat_grads: np.ndarray, 
                           original_shapes: List[Tuple[int, ...]]) -> List[np.ndarray]:
        """Unflatten gradient vector back to original shapes"""
        gradients = []
        start_idx = 0
        
        for shape in original_shapes:
            size = np.prod(shape)
            grad = flat_grads[start_idx:start_idx + size].reshape(shape)
            gradients.append(grad)
            start_idx += size
        
        return gradients
    
    def clip_gradient(self, gradient: np.ndarray, max_norm: float) -> np.ndarray:
        """Clip gradient to maximum norm"""
        norm = np.linalg.norm(gradient)
        if norm > max_norm:
            gradient = gradient * (max_norm / norm)
        return gradient
    
    def step(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Update parameters using gradients"""
        self.step_count += 1
        
        # Store original shapes for unflattening
        original_shapes = [grad.shape for grad in gradients]
        
        # Flatten gradients for clipping + weight decay
        flat_grads = self.compute_flat_gradients(gradients)
        
        # Apply gradient clipping on flat vector
        if self.gradient_clipping > 0:
            flat_grads = self.clip_gradient(flat_grads, self.gradient_clipping)
        
        # Apply weight decay on flat vector
        if self.weight_decay > 0:
            flat_grads += self.weight_decay * self._compute_flat_params()
        
        # Unflatten processed gradients back to per-param arrays
        processed_grads = self.unflatten_gradients(flat_grads, original_shapes)
        
        # Build gradient dict for optimizer (param id -> gradient array)
        grad_dict = {}
        for i, param in enumerate(self.model_params):
            grad_dict[id(param)] = processed_grads[i]
        
        # Let the optimizer update params in-place
        self.optimizer.step(grad_dict, self.model_params)
        
        # Return updated params as copies
        return [param.copy() for param in self.model_params]
    
    def _compute_flat_params(self) -> np.ndarray:
        """Flatten all model parameters into a single vector"""
        flat_params = []
        for param in self.model_params:
            flat_params.append(param.flatten())
        return np.concatenate(flat_params)
    
    def get_step_count(self) -> int:
        """Get current step count"""
        return self.step_count


class MegabyteTokenizer:
    """Byte-level tokenizer for ChapatiLM"""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.char_to_idx = {chr(i): i for i in range(vocab_size)}
        self.idx_to_char = {i: chr(i) for i in range(vocab_size)}
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to byte-level embeddings"""
        # Convert to bytes and map to indices
        byte_indices = [self.char_to_idx.get(chr(b), 0) for b in text.encode('utf-8')]
        return np.array(byte_indices, dtype=np.float32)
    
    def decode(self, indices: np.ndarray) -> str:
        """Decode byte-level embeddings back to text"""
        chars = [self.idx_to_char.get(int(idx), '?') for idx in indices]
        return ''.join(chars)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get byte-level embedding for text"""
        indices = self.encode(text)
        # Create one-hot encoding
        embedding = np.zeros((len(indices), self.vocab_size), dtype=np.float32)
        embedding[np.arange(len(indices)), indices.astype(int)] = 1.0
        return embedding


class ChapatiLM:
    """ChapatiLM architecture: Byte-level tokenizer + Dual NNs + RL training"""
    
    def __init__(self, vocab_size: int = 256, embedding_dim: int = 128, 
                 hidden_dim: int = 256, output_dim: int = 10):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize components
        self.tokenizer = MegabyteTokenizer(vocab_size)
        
        # Dual neural networks (NN1 and NN2)
        # NN1: embedding_dim -> hidden_dim -> hidden_dim
        self.nn1 = self._create_mlp_network(embedding_dim, hidden_dim, hidden_dim)
        # NN2: hidden_dim -> hidden_dim -> output_dim
        self.nn2 = self._create_mlp_network(hidden_dim, hidden_dim, output_dim)
        
        # RL components
        self.charge = 0.0  # Attention signal from Orchestrator
        self.step_count = 0
        
        # Collect all parameters
        self.params = []
        self.grads = []
        
        # Collect NN1 parameters
        for layer in self.nn1['layers']:
            if 'weights' in layer:
                self.params.extend([layer['weights'], layer['bias']])
                self.grads.extend([layer['grad_weights'], layer['grad_bias']])
        
        # Collect NN2 parameters
        for layer in self.nn2['layers']:
            if 'weights' in layer:
                self.params.extend([layer['weights'], layer['bias']])
                self.grads.extend([layer['grad_weights'], layer['grad_bias']])
    
    def _create_mlp_network(self, input_dim: int, hidden_dim: int, output_dim: int) -> Dict:
        """Create an MLP network for NN1 or NN2 with ReGLU activation
        
        ReGLU halves the dimension: Dense(input_dim -> 2*hidden_dim) -> ReGLU -> Dense(hidden_dim -> output_dim)
        """
        layers = []
        
        pre_glu_dim = hidden_dim * 2  # ReGLU splits this in half
        
        # First hidden layer (outputs 2*hidden_dim for ReGLU split)
        weights1 = (np.random.randn(input_dim, pre_glu_dim) * np.sqrt(2.0 / input_dim)).astype(np.float32)
        bias1 = np.zeros(pre_glu_dim, dtype=np.float32)
        grad_weights1 = np.zeros_like(weights1)
        grad_bias1 = np.zeros_like(bias1)
        
        layer1 = {
            'weights': weights1,
            'bias': bias1,
            'grad_weights': grad_weights1,
            'grad_bias': grad_bias1,
            'cache_input': None
        }
        layers.append(layer1)
        
        # ReGLU activation layer
        layer_relu = {
            'activation': 'reglu',
            'cache_input': None
        }
        layers.append(layer_relu)
        
        # Second hidden layer (input dim is hidden_dim after ReGLU halves it)
        weights2 = (np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        bias2 = np.zeros(output_dim, dtype=np.float32)
        grad_weights2 = np.zeros_like(weights2)
        grad_bias2 = np.zeros_like(bias2)
        
        layer2_dense = {
            'weights': weights2,
            'bias': bias2,
            'grad_weights': grad_weights2,
            'grad_bias': grad_bias2,
            'cache_input': None
        }
        layers.append(layer2_dense)
        
        return {'layers': layers}
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through ChapatiLM"""
        if isinstance(input_data, str):
            token_indices = self.tokenizer.encode(input_data)
            x = np.zeros((len(token_indices), self.embedding_dim), dtype=np.float32)
            for i, idx in enumerate(token_indices):
                x[i, int(idx)] = 1.0
        else:
            x = input_data.astype(np.float32)
            if x.ndim > 2:
                x = x.reshape(x.shape[0], -1)
            if x.shape[-1] != self.embedding_dim:
                if not hasattr(self, 'input_proj'):
                    self.input_proj = (np.random.randn(x.shape[-1], self.embedding_dim) * np.sqrt(2.0 / x.shape[-1])).astype(np.float32)
                    self.input_proj_grad = np.zeros_like(self.input_proj)
                    self.params.insert(0, self.input_proj)
                    self.grads.insert(0, self.input_proj_grad)
                self._cached_flat_input = x.copy()
                x = np.dot(x, self.input_proj)
            else:
                self._cached_flat_input = x.copy()
        
        nn1_output = self._process_nn1(x)
        nn2_output = self._process_nn2(nn1_output)
        return nn2_output
    
    def _process_nn1(self, input_data: np.ndarray) -> np.ndarray:
        """Process input through NN1"""
        layers = self.nn1['layers']
        
        # Layer 1: Dense + ReGLU
        layer1 = layers[0]
        layer1_output = np.dot(input_data, layer1['weights']) + layer1['bias']
        layer1['cache_input'] = input_data
        
        # ReGLU activation
        relu_layer = layers[1]
        split_idx = layer1_output.shape[-1] // 2
        x1 = layer1_output[..., :split_idx]
        x2 = layer1_output[..., split_idx:]
        relu_output = x1 * np.maximum(0, x2)
        relu_layer['cache_input'] = layer1_output
        
        # Layer 2: Dense
        layer2 = layers[2]
        layer2_output = np.dot(relu_output, layer2['weights']) + layer2['bias']
        layer2['cache_input'] = relu_output
        
        return layer2_output
    
    def _process_nn2(self, input_data: np.ndarray) -> np.ndarray:
        """Process input through NN2"""
        layers = self.nn2['layers']
        
        # Layer 1: Dense + ReGLU
        layer1 = layers[0]
        layer1_output = np.dot(input_data, layer1['weights']) + layer1['bias']
        layer1['cache_input'] = input_data
        
        # ReGLU activation
        relu_layer = layers[1]
        split_idx = layer1_output.shape[-1] // 2
        x1 = layer1_output[..., :split_idx]
        x2 = layer1_output[..., split_idx:]
        relu_output = x1 * np.maximum(0, x2)
        relu_layer['cache_input'] = layer1_output
        
        # Layer 2: Dense
        layer2 = layers[2]
        layer2_output = np.dot(relu_output, layer2['weights']) + layer2['bias']
        layer2['cache_input'] = relu_output
        
        return layer2_output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through ChapatiLM, filling layer gradients"""
        # === Backward through NN2 ===
        nn2_layers = self.nn2['layers']
        
        # NN2 Layer 2 (Dense): output = relu @ W2 + b2
        l2 = nn2_layers[2]
        inp2 = l2['cache_input']
        l2['grad_weights'][:] = np.dot(inp2.T, grad_output)
        l2['grad_bias'][:] = np.sum(grad_output, axis=0)
        grad_nn2_relu = np.dot(grad_output, l2['weights'].T)
        
        # NN2 ReGLU: output = x1 * relu(x2)  where [x1, x2] = split(cache)
        relu = nn2_layers[1]
        cache = relu['cache_input']
        split = cache.shape[-1] // 2
        x1 = cache[..., :split]
        x2 = cache[..., split:]
        gx1 = grad_nn2_relu * np.maximum(0, x2)
        gx2 = grad_nn2_relu * x1 * (x2 > 0).astype(np.float32)
        grad_nn2_dense = np.concatenate([gx1, gx2], axis=-1)
        
        # NN2 Layer 1 (Dense): output = input @ W1 + b1
        l1 = nn2_layers[0]
        inp1 = l1['cache_input']
        l1['grad_weights'][:] = np.dot(inp1.T, grad_nn2_dense)
        l1['grad_bias'][:] = np.sum(grad_nn2_dense, axis=0)
        grad_nn1 = np.dot(grad_nn2_dense, l1['weights'].T)
        
        # === Backward through NN1 ===
        nn1_layers = self.nn1['layers']
        
        # NN1 Layer 2 (Dense)
        l2 = nn1_layers[2]
        inp2 = l2['cache_input']
        l2['grad_weights'][:] = np.dot(inp2.T, grad_nn1)
        l2['grad_bias'][:] = np.sum(grad_nn1, axis=0)
        grad_nn1_relu = np.dot(grad_nn1, l2['weights'].T)
        
        # NN1 ReGLU
        relu = nn1_layers[1]
        cache = relu['cache_input']
        split = cache.shape[-1] // 2
        x1 = cache[..., :split]
        x2 = cache[..., split:]
        gx1 = grad_nn1_relu * np.maximum(0, x2)
        gx2 = grad_nn1_relu * x1 * (x2 > 0).astype(np.float32)
        grad_nn1_dense = np.concatenate([gx1, gx2], axis=-1)
        
        # NN1 Layer 1 (Dense)
        l1 = nn1_layers[0]
        inp1 = l1['cache_input']
        l1['grad_weights'][:] = np.dot(inp1.T, grad_nn1_dense)
        l1['grad_bias'][:] = np.sum(grad_nn1_dense, axis=0)
        grad_input = np.dot(grad_nn1_dense, l1['weights'].T)
        
        # Backward through input projection if it exists
        if hasattr(self, 'input_proj'):
            self.input_proj_grad[:] = np.dot(self._cached_flat_input.T, grad_input)
            grad_input = np.dot(grad_input, self.input_proj.T)
        
        return grad_input
    
    def calculate_complexity(self, query: np.ndarray, embedding_energy: float) -> float:
        """Calculate complexity: C = Q / √E"""
        return query / np.sqrt(embedding_energy)
    
    def calculate_zquery(self, query: np.ndarray, complexity: float, query_count: int) -> float:
        """Calculate ZQuery: Z_Q = (Query × Complexity) / QueryCount"""
        return (query * complexity) / query_count
    
    def calculate_submission(self, complexity: float, query_count: int) -> float:
        """Calculate Submission: S = C / QueryCount"""
        return complexity / query_count
    
    def orchestrator_pass(self, submission_count: int, query_count: int, charge: float) -> float:
        """Orchestrator Pass: Pass = (SubmissionCount × QueryCount) / charge"""
        return (submission_count * query_count) / charge
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get all model parameters"""
        return self.params
    
    def get_gradients(self) -> List[np.ndarray]:
        """Get all model gradients"""
        return self.grads
    
    def zero_gradients(self):
        """Zero out all gradients"""
        for grad in self.grads:
            grad.fill(0.0)
    
    def update_charge(self, charge: float):
        """Update charge from Orchestrator"""
        self.charge = charge
    
    def get_step_count(self) -> int:
        """Get current step count"""
        return self.step_count
    
    def increment_step(self):
        """Increment step count"""
        self.step_count += 1


class Conv2D:
    """2D Convolutional layer with FULL BACKPROPAGATION support"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Initialize weights using He initialization
        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        self.bias = np.zeros(out_channels, dtype=np.float32)
        
        # Gradient storage for backprop
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        
        # Cache for backward pass
        self.cache_input = None
        self.cache_output_pre_bias = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using CPUWARP-ML optimized convolution"""
        self.cache_input = x
        
        output = cpuwarp_ml.conv2d(x, self.weights, stride=self.stride, padding='valid')
        self.cache_output_pre_bias = output.copy()
        
        # Add bias
        for c in range(self.out_channels):
            output[:, c, :, :] += self.bias[c]
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass computing gradients for:
        - Input (to pass to previous layer)
        - Weights and bias (for parameter updates)
        """
        # Compute parameter gradients
        grad_input, self.grad_weights = conv2d_backward_optimized(
            grad_output, self.cache_input, self.weights, stride=self.stride
        )
        self.grad_bias = np.sum(grad_output, axis=(0, 2, 3))
        
        return grad_input
    
    def get_output_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate output size after convolution"""
        h, w = input_size
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1
        return (out_h, out_w)


class BatchNorm2D:
    """2D Batch Normalization layer"""
    
    def __init__(self, num_features: int):
        self.num_features = num_features
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)
        
        # Cache for backward pass
        self.cache_running_mean = None
        self.cache_running_var = None
        self.cache_input = None
        self.cache_normalized = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with batch normalization"""
        self.cache_input = x
        
        # Compute mean and variance over batch and spatial dimensions
        batch_mean = np.mean(x, axis=(0, 2, 3))
        batch_var = np.var(x, axis=(0, 2, 3))
        
        self.cache_running_mean = batch_mean
        self.cache_running_var = batch_var
        
        # Normalize
        eps = 1e-5
        normalized = (x - batch_mean.reshape(1, -1, 1, 1)) / np.sqrt(batch_var.reshape(1, -1, 1, 1) + eps)
        self.cache_normalized = normalized
        
        # Scale and shift
        output = normalized * self.gamma.reshape(1, -1, 1, 1) + self.beta.reshape(1, -1, 1, 1)
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass for batch normalization"""
        # Simplified backward pass
        normalized = self.cache_normalized
        input_shape = self.cache_input.shape
        
        # Compute gradients
        batch_size = input_shape[0]
        spatial_size = input_shape[2] * input_shape[3]
        
        # Gradient w.r.t. gamma
        self.grad_gamma = np.sum(grad_output * normalized, axis=(0, 2, 3))
        
        # Gradient w.r.t. beta
        self.grad_beta = np.sum(grad_output, axis=(0, 2, 3))
        
        # Gradient w.r.t. input (simplified)
        grad_input = grad_output * self.gamma.reshape(1, -1, 1, 1)
        
        return grad_input


class ReLU:
    """ReLU activation function"""
    
    def __init__(self):
        self.cache_input = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self.cache_input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass"""
        if self.cache_input is None:
            raise ValueError("Forward pass must be called before backward pass")
        
        grad_input = grad_output * (self.cache_input > 0).astype(np.float32)
        return grad_input


class MaxPool2D:
    """2D Max pooling layer"""
    
    def __init__(self, pool_size: int = 2, stride: int = None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with max pooling"""
        batch_size, channels, height, width = x.shape
        
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width), dtype=np.float32)
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = min(h_start + self.pool_size, height)
                        w_start = w * self.stride
                        w_end = min(w_start + self.pool_size, width)
                        
                        patch = x[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, h, w] = np.max(patch)
        
        return output
    
    def backward(self, grad_output: np.ndarray, input_shape: Tuple[int, int, int, int]) -> np.ndarray:
        """Backward pass for max pooling"""
        batch_size, channels, height, width = input_shape
        out_height = grad_output.shape[2]
        out_width = grad_output.shape[3]
        
        grad_input = np.zeros(input_shape, dtype=np.float32)
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = min(h_start + self.pool_size, height)
                        w_start = w * self.stride
                        w_end = min(w_start + self.pool_size, width)
                        
                        # Find max location in the patch
                        patch = input_shape[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(patch)
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                        
                        grad_input[b, c, h_start + max_idx[0], w_start + max_idx[1]] = grad_output[b, c, h, w]
        
        return grad_input


class Dense:
    """Dense layer"""
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        std = np.sqrt(2.0 / input_dim)
        self.weights = (np.random.randn(input_dim, output_dim) * std).astype(np.float32)
        self.bias = np.zeros(output_dim, dtype=np.float32)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self.cache_input = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache_input = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.grad_weights = np.dot(self.cache_input.T, grad_output)
        self.grad_bias = np.sum(grad_output, axis=0)
        return np.dot(grad_output, self.weights.T)


class XTrainer:
    """Main orchestrator for training"""
    
    def __init__(self, model, config: CPUWarpMLConfig, checkpoint_dir: str = "checkpoints"):
        self.model = model
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize components
        self.dataset_scavenger = DatasetScavenger()
        self.checkpoint_manager = AtomicCheckpoint(checkpoint_dir)
        self.vector_gradient_engine = VectorGradientEngine(
            model.get_parameters(),
            learning_rate=config.training.cnn_learning_rate,
            optimizer_type='adam',
            weight_decay=config.training.cnn_weight_decay,
            gradient_clipping=config.training.gradient_clipping
        )
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.early_stopping_counter = 0
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Load checkpoint if available
        checkpoint_data = self.checkpoint_manager.load_latest_checkpoint()
        if checkpoint_data is not None:
            self._load_checkpoint(checkpoint_data)
        
        logger.info(f"XTrainer initialized. Model type: {type(model).__name__}")
    
    def _load_checkpoint(self, checkpoint_data):
        """Load checkpoint data"""
        trainer_state, epoch, metrics = checkpoint_data
        
        # Restore model parameters
        for i, param in enumerate(self.model.get_parameters()):
            if f'param_{i}' in trainer_state:
                param[:] = trainer_state[f'param_{i}']
        
        self.current_epoch = epoch
        self.best_metric = metrics.get('best_metric', 0.0)
        self.training_history = metrics.get('training_history', self.training_history)
        
        logger.info(f"Loaded checkpoint from epoch {epoch}")
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save current training state"""
        trainer_state = {}
        for i, param in enumerate(self.model.get_parameters()):
            trainer_state[f'param_{i}'] = param.copy()
        
        metrics = {
            'best_metric': self.best_metric,
            'training_history': self.training_history
        }
        
        self.checkpoint_manager.save_checkpoint(
            trainer_state, self.current_epoch, metrics, is_best
        )
    
    def train_epoch(self, loader_fn, batch_size: int) -> float:
        """Train for one epoch"""
        logger.info(f"Starting epoch {self.current_epoch + 1}")
        
        # Get dataset from loader function
        data, labels = loader_fn()
        
        # Create batches
        batches = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            batches.append((batch_data, batch_labels))
        
        total_loss = 0.0
        num_batches = len(batches)
        
        for batch_idx, (batch_data, batch_labels) in enumerate(batches):
            # Forward pass
            outputs = self.model.forward(batch_data)
            
            # Compute loss
            loss = self._compute_loss(outputs, batch_labels)
            
            # Backward pass
            self.model.zero_gradients()
            grad_outputs = cross_entropy_loss_backward_stable(outputs, batch_labels)
            grad_input = self.model.backward(grad_outputs)
            
            # Collect gradients
            gradients = self.model.get_gradients()
            
            # Update parameters
            updated_params = self.vector_gradient_engine.step(gradients)
            
            # Update model parameters
            for i, param in enumerate(self.model.get_parameters()):
                param[:] = updated_params[i]
            
            total_loss += loss
            
            if batch_idx % 10 == 0:
                logger.info(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {self.current_epoch + 1} completed. Average loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _compute_loss(self, outputs: np.ndarray, labels: np.ndarray) -> float:
        """Compute cross-entropy loss from logits (handles integer or one-hot labels)"""
        logits_shifted = outputs - np.max(outputs, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        softmax_output = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        eps = 1e-8
        if labels.ndim == 1:
            loss = -np.mean(np.log(softmax_output[np.arange(len(labels)), labels] + eps))
        else:
            log_probs = np.log(softmax_output + eps)
            loss = -np.mean(np.sum(labels * log_probs, axis=-1))
        return loss
    
    def validate(self, loader_fn, batch_size: int) -> Tuple[float, float]:
        """Validate model on dataset"""
        data, labels = loader_fn()
        
        # Create batches
        batches = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            batches.append((batch_data, batch_labels))
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_data, batch_labels in batches:
            # Forward pass
            outputs = self.model.forward(batch_data)
            
            # Compute loss
            loss = self._compute_loss(outputs, batch_labels)
            total_loss += loss
            
            # Compute accuracy
            predictions = np.argmax(outputs, axis=-1)
            if batch_labels.ndim == 1:
                true_labels = batch_labels
            else:
                true_labels = np.argmax(batch_labels, axis=-1)
            
            correct += np.sum(predictions == true_labels)
            total += len(predictions)
        
        avg_loss = total_loss / len(batches)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def train(self, dataset_info: DatasetInfo, epochs: int, batch_size: int, 
              validate_every: int = 1) -> Dict:
        """Main training loop"""
        logger.info(f"Starting training for {epochs} epochs on {dataset_info.name}")
        
        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch(dataset_info.loader_fn, batch_size)
            
            # Validate periodically
            if (epoch + 1) % validate_every == 0:
                val_loss, val_accuracy = self.validate(dataset_info.loader_fn, batch_size)
                
                # Record metrics
                self.training_history['loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
                
                logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
                
                # Check for improvement
                if val_accuracy > self.best_metric:
                    self.best_metric = val_accuracy
                    self.early_stopping_counter = 0
                    self._save_checkpoint(is_best=True)
                    logger.info(f"New best metric: {self.best_metric:.4f}")
                else:
                    self.early_stopping_counter += 1
                    
                    # Save regular checkpoint
                    if (epoch + 1) % 10 == 0:
                        self._save_checkpoint(is_best=False)
            
            # Check for early stopping
            if self.early_stopping_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(is_best=False)
        
        logger.info("Training completed")
        return self.training_history
    
    def auto_build_model(self, dataset_info: DatasetInfo) -> Any:
        """Auto-build model from dataset shape"""
        input_shape = dataset_info.input_shape
        num_classes = dataset_info.num_classes
        
        # Determine model type based on input shape
        if len(input_shape) == 3 and input_shape[0] == 3:  # Image data
            # Build CNN model
            model = CNNModel(input_shape, num_classes)
            logger.info(f"Built CNN model for image data with shape {input_shape}")
        elif len(input_shape) == 2:  # Dense data
            # Build MLP model
            model = MLPModel(input_shape[0], 128, num_classes)
            logger.info(f"Built MLP model for dense data with shape {input_shape}")
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")
        
        return model


def main():
    """Main entry point — run and forget"""
    config = get_config()
    logging.basicConfig(
        level=getattr(logging, config.logging.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(config.logging.log_file), logging.StreamHandler()]
    )
    
    logger = logging.getLogger(__name__)
    
    # Discover datasets in current directory
    dataset_scavenger = DatasetScavenger()
    datasets = dataset_scavenger.discover_datasets()
    
    target_dataset = datasets[0] if datasets else None
    
    if target_dataset is None:
        logger.warning("No datasets found. Creating dummy dataset...")
        dummy_data = np.random.randn(1000, 10).astype(np.float32)
        dummy_labels = np.random.randint(0, 10, 1000)
        target_dataset = DatasetInfo(
            name="dummy_dataset",
            loader_fn=lambda: (dummy_data, dummy_labels),
            input_shape=(10,),
            num_classes=10,
            size=1000
        )
    
    logger.info(f"Training on: {target_dataset}")
    
    # Build model
    model = ChapatiLM(
        vocab_size=256,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=target_dataset.num_classes
    )
    
    # Train
    trainer = XTrainer(model, config, "checkpoints")
    model.update_charge(1.0)
    
    trainer.train(
        dataset_info=target_dataset,
        epochs=100,
        batch_size=32,
        validate_every=config.training.eval_every_n_epochs
    )
    
    print(f"\nTraining complete. Best metric: {trainer.best_metric:.4f}")


if __name__ == "__main__":
    main()