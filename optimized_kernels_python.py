
import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True, fastmath=True)
def optimized_matmul(A, B, C, M, K, N):
    """
    Optimized matrix multiplication using Numba for JIT compilation and parallelism.
    C = A @ B
    A: M x K
    B: K x N
    C: M x N
    """
    # Numba's parallel=True handles the parallelism similar to OpenMP
    # Numba's JIT compilation and fastmath=True will attempt SIMD optimizations

    # Cache-friendly block sizes (can be tuned, Numba might optimize this itself)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 256
    BLOCK_SIZE_N = 64
    
    for bi in prange(0, M, BLOCK_SIZE_M):
        for bj in prange(0, N, BLOCK_SIZE_N):
            for bk in prange(0, K, BLOCK_SIZE_K):
                end_i = min(bi + BLOCK_SIZE_M, M)
                end_j = min(bj + BLOCK_SIZE_N, N)
                end_k = min(bk + BLOCK_SIZE_K, K)

                for i in range(bi, end_i):
                    for j in range(bj, end_j):
                        # If it's the first block_k, initialize C[i, j] to 0
                        # Otherwise, add to existing C[i, j]
                        if bk == 0:
                            C[i, j] = 0.0
                        for k in range(bk, end_k):
                            C[i, j] += A[i, k] * B[k, j]


@jit(nopython=True, parallel=True, fastmath=True)
def optimized_conv2d(input_array, kernel_array, output_array,
                   batch_size, in_channels, in_height, in_width,
                   out_channels, kernel_height, kernel_width):
    out_height = in_height - kernel_height + 1
    out_width = in_width - kernel_width + 1

    for b in prange(batch_size):
        for oc in prange(out_channels):
            for oh in range(out_height):
                for ow in range(out_width):
                    sum_val = 0.0
                    for ic in range(in_channels):
                        for kh in range(kernel_height):
                            for kw in range(kernel_width):
                                in_h = oh + kh
                                in_w = ow + kw
                                if 0 <= in_h < in_height and 0 <= in_w < in_width:
                                    sum_val += input_array[b, ic, in_h, in_w] * kernel_array[oc, ic, kh, kw]
                    output_array[b, oc, oh, ow] = sum_val


@jit(nopython=True, parallel=True, fastmath=True)
def optimized_relu(input_array, output_array, size):
    for i in prange(size):
        output_array[i] = max(0.0, input_array[i])


@jit(nopython=True, parallel=True, fastmath=True)
def optimized_softmax(input_array, output_array, size):
    # For numerical stability, subtract max from input
    max_val = -np.inf # Initialize with negative infinity
    for i in range(size):
        if input_array[i] > max_val:
            max_val = input_array[i]

    exp_sum = 0.0
    for i in prange(size):
        output_array[i] = np.exp(input_array[i] - max_val)
        exp_sum += output_array[i]

    for i in prange(size):
        output_array[i] /= exp_sum


@jit(nopython=True, parallel=True, fastmath=True)
def optimized_add(a, b, c, size):
    for i in prange(size):
        c[i] = a[i] + b[i]


@jit(nopython=True, parallel=True, fastmath=True)
def optimized_mul(a, b, c, size):
    for i in prange(size):
        c[i] = a[i] * b[i]


@jit(nopython=True, parallel=True, fastmath=True)
def optimized_layer_norm(input_array, output_array, gamma, beta,
                                batch_size, features, epsilon):
    for b in prange(batch_size):
        x = input_array[b * features : (b + 1) * features]
        y = output_array[b * features : (b + 1) * features]

        mean = np.mean(x)
        variance = np.var(x)
        
        inv_std = 1.0 / np.sqrt(variance + epsilon)

        for i in range(features):
            norm = (x[i] - mean) * inv_std
            y[i] = norm * gamma[i] + beta[i]


# Utility functions (aligned_malloc, aligned_free, detect_cpu_features, get_cpu_features, benchmark_matmul)
# These are more C-specific and less relevant for a Numba/Python implementation.
# Memory management is handled by Python's garbage collector and NumPy/Numba's internal mechanisms.
# CPU feature detection is often handled implicitly by Numba based on the target CPU.
# Benchmarking would typically be done using Python's `time` module or dedicated benchmarking libraries.
