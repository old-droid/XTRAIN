#!/usr/bin/env python3
"""
Test script to verify that CPUWARP-ML works correctly after C extension removal
"""

import numpy as np
import cpuwarp_ml
import time

def test_basic_functionality():
    """Test basic mathematical operations"""
    print("Testing basic functionality...")
    
    # Test matrix multiplication
    A = np.random.randn(100, 100).astype(np.float32)
    B = np.random.randn(100, 100).astype(np.float32)
    
    start = time.time()
    C = cpuwarp_ml.matmul(A, B)
    warp_time = time.time() - start
    
    # Verify result is correct
    expected = np.dot(A, B)
    error = np.mean(np.abs(C - expected))
    
    print(f"  Matrix multiplication: {warp_time:.4f}s, error: {error:.2e}")
    assert error < 1e-5, f"Matrix multiplication error too high: {error}"
    
    # Test ReLU
    x = np.random.randn(1000).astype(np.float32)
    result = cpuwarp_ml.relu(x)
    expected = np.maximum(0, x)
    error = np.mean(np.abs(result - expected))
    print(f"  ReLU: error: {error:.2e}")
    assert error < 1e-6, f"ReLU error too high: {error}"
    
    # Test softmax
    x = np.random.randn(50).astype(np.float32)
    result = cpuwarp_ml.softmax(x)
    expected = np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    error = np.mean(np.abs(result - expected))
    print(f"  Softmax: error: {error:.2e}")
    assert error < 1e-6, f"Softmax error too high: {error}"
    
    print("[OK] All basic functionality tests passed!")

def test_performance_stats():
    """Test performance statistics"""
    print("\nTesting performance statistics...")
    
    stats = cpuwarp_ml.cpuwarp.get_performance_stats()
    
    print(f"  CPU Vendor: {stats['cpu_info']['vendor']}")
    print(f"  CPU Cores: {stats['cpu_info']['cores']}")
    print(f"  CPU Threads: {stats['cpu_info']['threads']}")
    print(f"  CPU Features: {', '.join(stats['cpu_info']['features'])}")
    print(f"  C Extensions: {stats['c_extensions']}")
    
    assert stats['c_extensions'] == False, "C extensions should be disabled"
    print("[OK] Performance statistics test passed!")

def test_workload_analysis():
    """Test workload analysis functionality"""
    print("\nTesting workload analysis...")
    
    analyzer = cpuwarp_ml.WorkloadAnalyzer()
    
    # Test workload classification
    workload_type = analyzer.classify_workload('matmul', (100, 100))
    print(f"  Matrix multiplication classified as: {workload_type}")
    
    workload_type = analyzer.classify_workload('relu', (1000,))
    print(f"  ReLU classified as: {workload_type}")
    
    print("[OK] Workload analysis test passed!")

if __name__ == "__main__":
    print("CPUWARP-ML Cleaned Code Test")
    print("=" * 40)
    
    try:
        test_basic_functionality()
        test_performance_stats()
        test_workload_analysis()
        
        print("\n" + "=" * 40)
        print("ALL TESTS PASSED!")
        print("C extensions have been successfully removed.")
        print("The framework now uses pure Python/NumPy with Numba acceleration.")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        raise