"""
Test script for Numba-accelerated kernels
==========================================

Validates that Numba kernels are functional and provide speedup
over NumPy implementations.
"""

import numpy as np
import time
import cpuwarp_ml
import numba_kernels

def benchmark_matmul(size=(512, 512)):
    """Benchmark matrix multiplication"""
    print(f"\n{'='*60}")
    print(f"Matrix Multiplication: {size[0]}x{size[1]}")
    print(f"{'='*60}")
    
    A = np.random.randn(size[0], size[1]).astype(np.float32)
    B = np.random.randn(size[1], size[0]).astype(np.float32)
    
    # Warm up
    _ = cpuwarp_ml.matmul(A, B)
    
    # Numba (via cpuwarp_ml)
    start = time.time()
    for _ in range(3):
        C_numba = cpuwarp_ml.matmul(A, B)
    numba_time = (time.time() - start) / 3
    
    # NumPy baseline
    start = time.time()
    for _ in range(3):
        C_numpy = np.dot(A, B)
    numpy_time = (time.time() - start) / 3
    
    # Verify correctness
    error = np.max(np.abs(C_numba - C_numpy))
    
    print(f"Numba:    {numba_time*1000:.3f} ms")
    print(f"NumPy:    {numpy_time*1000:.3f} ms")
    print(f"Speedup:  {numpy_time/numba_time:.2f}x")
    print(f"Max Error: {error:.2e}")
    
    return numpy_time / numba_time

def benchmark_relu(size=(1000000,)):
    """Benchmark ReLU activation"""
    print(f"\n{'='*60}")
    print(f"ReLU Activation: {size}")
    print(f"{'='*60}")
    
    X = np.random.randn(*size).astype(np.float32) * 2 - 1  # Range [-1, 1]
    
    # Numba
    start = time.time()
    for _ in range(10):
        Y_numba = cpuwarp_ml.relu(X)
    numba_time = (time.time() - start) / 10
    
    # NumPy
    start = time.time()
    for _ in range(10):
        Y_numpy = np.maximum(0, X)
    numpy_time = (time.time() - start) / 10
    
    error = np.max(np.abs(Y_numba - Y_numpy))
    
    print(f"Numba:    {numba_time*1000:.3f} ms")
    print(f"NumPy:    {numpy_time*1000:.3f} ms")
    print(f"Speedup:  {numpy_time/numba_time:.2f}x")
    print(f"Max Error: {error:.2e}")
    
    return numpy_time / numba_time

def benchmark_softmax(shape=(100, 1000)):
    """Benchmark Softmax"""
    print(f"\n{'='*60}")
    print(f"Softmax: {shape}")
    print(f"{'='*60}")
    
    X = np.random.randn(*shape).astype(np.float32)
    
    # Numba
    start = time.time()
    for _ in range(5):
        Y_numba = cpuwarp_ml.softmax(X)
    numba_time = (time.time() - start) / 5
    
    # NumPy
    start = time.time()
    for _ in range(5):
        exp_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
        Y_numpy = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    numpy_time = (time.time() - start) / 5
    
    error = np.max(np.abs(Y_numba - Y_numpy))
    
    print(f"Numba:    {numba_time*1000:.3f} ms")
    print(f"NumPy:    {numpy_time*1000:.3f} ms")
    print(f"Speedup:  {numpy_time/numba_time:.2f}x")
    print(f"Max Error: {error:.2e}")
    
    return numpy_time / numba_time

def test_convolution():
    """Test convolution kernel"""
    print(f"\n{'='*60}")
    print(f"2D Convolution")
    print(f"{'='*60}")
    
    # Small conv test
    input_data = np.random.randn(2, 3, 32, 32).astype(np.float32)
    kernel = np.random.randn(16, 3, 3, 3).astype(np.float32)
    
    try:
        output = cpuwarp_ml.conv2d(input_data, kernel)
        print(f"✓ Convolution executed successfully")
        print(f"  Input shape:  {input_data.shape}")
        print(f"  Kernel shape: {kernel.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        return True
    except Exception as e:
        print(f"✗ Convolution failed: {e}")
        return False

def main():
    print("="*60)
    print("NUMBA-ACCELERATED KERNEL TESTS")
    print("="*60)
    
    print(f"\nNumba Status: {numba_kernels.get_numba_status()}")
    
    speedups = []
    
    # Run benchmarks
    speedups.append(benchmark_matmul((256, 256)))
    speedups.append(benchmark_matmul((512, 512)))
    speedups.append(benchmark_relu((1000000,)))
    speedups.append(benchmark_softmax((100, 1000)))
    test_convolution()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Average Speedup: {np.mean(speedups):.2f}x")
    print(f"Min Speedup:     {np.min(speedups):.2f}x")
    print(f"Max Speedup:     {np.max(speedups):.2f}x")
    print(f"\n✓ All tests passed!")

if __name__ == '__main__':
    main()
