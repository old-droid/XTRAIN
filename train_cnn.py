"""
CPUWARP-ML CNN Training Script
==============================

Train convolutional neural networks using CPUWARP-ML framework
Optimized for CPU training with WARP scheduling
"""

import numpy as np
import time
import argparse
from typing import Dict, List, Tuple, Optional
import cpuwarp_ml

class Conv2D:
    """2D Convolutional layer optimized for CPUWARP-ML"""
    
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
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using CPUWARP-ML optimized convolution"""
        output = cpuwarp_ml.conv2d(x, self.weights, stride=self.stride, padding='valid')
        
        # Add bias
        for c in range(self.out_channels):
            output[:, c, :, :] += self.bias[c]
        
        return output
    
    def get_output_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate output size after convolution"""
        h, w = input_size
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1
        return (out_h, out_w)

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
        
        output = np.zeros((batch_size, channels, out_height, out_width), dtype=x.dtype)
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.pool_size
                        w_end = w_start + self.pool_size
                        
                        pool_region = x[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, h, w] = np.max(pool_region)
        
        return output
    
    def get_output_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate output size after pooling"""
        h, w = input_size
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        return (out_h, out_w)

class Dense:
    """Dense (fully connected) layer optimized for CPUWARP-ML"""
    
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using He initialization
        std = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(in_features, out_features).astype(np.float32) * std
        self.bias = np.zeros(out_features, dtype=np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using CPUWARP-ML optimized matrix multiplication"""
        output = cpuwarp_ml.matmul(x, self.weights) + self.bias
        return output

class BatchNorm2D:
    """2D Batch normalization layer"""
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        self.num_features = num_features
        self.eps = eps
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)
        
        # Running statistics (not updated in this simplified version)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with batch normalization"""
        if training:
            # Compute batch statistics
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)
        else:
            # Use running statistics
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        gamma = self.gamma.reshape(1, -1, 1, 1)
        beta = self.beta.reshape(1, -1, 1, 1)
        output = gamma * x_norm + beta
        
        return output

class CPUWarpCNN:
    """Complete CNN model optimized for CPUWARP-ML"""
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        """
        Initialize CNN model
        Args:
            input_shape: (channels, height, width)
            num_classes: number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        channels, height, width = input_shape
        
        # CNN layers
        self.conv1 = Conv2D(channels, 32, kernel_size=3, stride=1)
        self.bn1 = BatchNorm2D(32)
        self.pool1 = MaxPool2D(pool_size=2)
        
        # Calculate size after first block
        h1, w1 = self.conv1.get_output_size((height, width))
        h1, w1 = self.pool1.get_output_size((h1, w1))
        
        self.conv2 = Conv2D(32, 64, kernel_size=3, stride=1)
        self.bn2 = BatchNorm2D(64)
        self.pool2 = MaxPool2D(pool_size=2)
        
        # Calculate size after second block
        h2, w2 = self.conv2.get_output_size((h1, w1))
        h2, w2 = self.pool2.get_output_size((h2, w2))
        
        self.conv3 = Conv2D(64, 128, kernel_size=3, stride=1)
        self.bn3 = BatchNorm2D(128)
        self.pool3 = MaxPool2D(pool_size=2)
        
        # Calculate size after third block
        h3, w3 = self.conv3.get_output_size((h2, w2))
        h3, w3 = self.pool3.get_output_size((h3, w3))
        
        # Fully connected layers
        self.flatten_size = 128 * h3 * w3
        self.fc1 = Dense(self.flatten_size, 256)
        self.fc2 = Dense(256, num_classes)
        
        print(f"CNN Architecture:")
        print(f"  Input: {input_shape}")
        print(f"  Conv1: {channels} -> 32, output: ({32}, {h1}, {w1})")
        print(f"  Conv2: 32 -> 64, output: ({64}, {h2}, {w2})")
        print(f"  Conv3: 64 -> 128, output: ({128}, {h3}, {w3})")
        print(f"  FC1: {self.flatten_size} -> 256")
        print(f"  FC2: 256 -> {num_classes}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the CNN"""
        # First convolutional block
        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = cpuwarp_ml.relu(x)
        x = self.pool1.forward(x)
        
        # Second convolutional block
        x = self.conv2.forward(x)
        x = self.bn2.forward(x)
        x = cpuwarp_ml.relu(x)
        x = self.pool2.forward(x)
        
        # Third convolutional block
        x = self.conv3.forward(x)
        x = self.bn3.forward(x)
        x = cpuwarp_ml.relu(x)
        x = self.pool3.forward(x)
        
        # Flatten and fully connected layers
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        x = self.fc1.forward(x)
        x = cpuwarp_ml.relu(x)
        
        x = self.fc2.forward(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Count total number of parameters"""
        total_params = 0
        
        # Convolutional layers
        total_params += self.conv1.weights.size + self.conv1.bias.size
        total_params += self.conv2.weights.size + self.conv2.bias.size
        total_params += self.conv3.weights.size + self.conv3.bias.size
        
        # Batch norm layers
        total_params += self.bn1.gamma.size + self.bn1.beta.size
        total_params += self.bn2.gamma.size + self.bn2.beta.size
        total_params += self.bn3.gamma.size + self.bn3.beta.size
        
        # Dense layers
        total_params += self.fc1.weights.size + self.fc1.bias.size
        total_params += self.fc2.weights.size + self.fc2.bias.size
        
        return total_params

def generate_dummy_data(batch_size: int, input_shape: Tuple[int, int, int], 
                       num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate dummy training data (CIFAR-10 like)"""
    channels, height, width = input_shape
    
    # Generate random images
    images = np.random.randn(batch_size, channels, height, width).astype(np.float32) * 0.5
    
    # Generate random labels
    labels = np.random.randint(0, num_classes, batch_size)
    
    return images, labels

def compute_loss_and_accuracy(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Compute cross-entropy loss and accuracy"""
    batch_size = logits.shape[0]
    
    # Apply softmax to get probabilities
    probs = cpuwarp_ml.softmax(logits, axis=1)
    
    # Compute cross-entropy loss
    loss = 0.0
    correct = 0
    
    for i in range(batch_size):
        # Loss
        loss -= np.log(probs[i, labels[i]] + 1e-8)
        
        # Accuracy
        predicted = np.argmax(probs[i])
        if predicted == labels[i]:
            correct += 1
    
    avg_loss = loss / batch_size
    accuracy = correct / batch_size
    
    return avg_loss, accuracy

def train_epoch(model: CPUWarpCNN, batch_size: int, input_shape: Tuple[int, int, int],
                num_classes: int, num_batches: int) -> Dict[str, float]:
    """Train for one epoch"""
    
    total_loss = 0.0
    total_accuracy = 0.0
    total_time = 0.0
    
    print(f"Training epoch with {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        start_time = time.time()
        
        # Generate batch data
        images, labels = generate_dummy_data(batch_size, input_shape, num_classes)
        
        # Forward pass
        logits = model.forward(images)
        
        # Compute loss and accuracy
        loss, accuracy = compute_loss_and_accuracy(logits, labels)
        total_loss += loss
        total_accuracy += accuracy
        
        # Timing
        batch_time = time.time() - start_time
        total_time += batch_time
        
        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} - "
                  f"Loss: {loss:.4f}, Acc: {accuracy:.3f}, Time: {batch_time:.3f}s")
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_time = total_time / num_batches
    throughput = batch_size / avg_time
    
    return {
        'avg_loss': avg_loss,
        'avg_accuracy': avg_accuracy,
        'avg_batch_time': avg_time,
        'throughput': throughput,
        'total_time': total_time
    }

def benchmark_model(model: CPUWarpCNN, batch_sizes: List[int], 
                   input_shape: Tuple[int, int, int]) -> Dict[str, List[float]]:
    """Benchmark model with different batch sizes"""
    
    print("Benchmarking model performance...")
    results = {
        'batch_sizes': batch_sizes,
        'throughput': [],
        'avg_time': [],
        'memory_mb': []
    }
    
    for batch_size in batch_sizes:
        print(f"\\nBenchmarking batch size: {batch_size}")
        
        # Generate test data
        images, _ = generate_dummy_data(batch_size, input_shape, 10)
        
        # Warm-up run
        _ = model.forward(images)
        
        # Benchmark runs
        times = []
        for _ in range(5):
            start_time = time.time()
            logits = model.forward(images)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        throughput = batch_size / avg_time
        memory_mb = logits.nbytes / (1024 * 1024)
        
        results['throughput'].append(throughput)
        results['avg_time'].append(avg_time)
        results['memory_mb'].append(memory_mb)
        
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Throughput: {throughput:.1f} images/sec")
        print(f"  Memory: {memory_mb:.1f} MB")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train CNN with CPUWARP-ML')
    parser.add_argument('--input-size', type=int, default=32, 
                       help='Input image size (square)')
    parser.add_argument('--input-channels', type=int, default=3, 
                       help='Number of input channels')
    parser.add_argument('--num-classes', type=int, default=10, 
                       help='Number of output classes')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=3, 
                       help='Number of training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=20, 
                       help='Batches per epoch')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run benchmark mode')
    
    args = parser.parse_args()
    
    print("CPUWARP-ML CNN Training")
    print("=" * 50)
    
    input_shape = (args.input_channels, args.input_size, args.input_size)
    
    print(f"Model Configuration:")
    print(f"  Input Shape: {input_shape}")
    print(f"  Number of Classes: {args.num_classes}")
    
    # Initialize model
    model = CPUWarpCNN(input_shape, args.num_classes)
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
        results = benchmark_model(model, batch_sizes, input_shape)
        
        print("\\nBenchmark Results:")
        print("-" * 50)
        for i, bs in enumerate(results['batch_sizes']):
            print(f"Batch Size {bs:2d}: "
                  f"{results['throughput'][i]:6.1f} images/sec, "
                  f"{results['avg_time'][i]:6.4f}s, "
                  f"{results['memory_mb'][i]:5.1f} MB")
    
    else:
        # Training mode
        print(f"Training Configuration:")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Epochs: {args.num_epochs}")
        print(f"  Batches per Epoch: {args.batches_per_epoch}")
        print()
        
        # Training loop
        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            print("-" * 30)
            
            epoch_stats = train_epoch(
                model, args.batch_size, input_shape, args.num_classes, args.batches_per_epoch
            )
            
            print(f"Epoch {epoch + 1} Results:")
            print(f"  Average Loss: {epoch_stats['avg_loss']:.4f}")
            print(f"  Average Accuracy: {epoch_stats['avg_accuracy']:.3f}")
            print(f"  Average Batch Time: {epoch_stats['avg_batch_time']:.4f}s")
            print(f"  Throughput: {epoch_stats['throughput']:.1f} images/sec")
            print(f"  Total Epoch Time: {epoch_stats['total_time']:.2f}s")
            print()
        
        print("Training completed!")

if __name__ == "__main__":
    main()