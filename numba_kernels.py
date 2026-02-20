"""
Numba-accelerated kernels for CPUWARP-ML
=========================================

High-performance compiled kernels using Numba JIT compilation with parallelism.
Provides drop-in replacements for core ML operations with automatic CPU feature detection.

Author: CPUWARP-ML Numba Extension
License: MIT
"""

import numpy as np
import warnings
from typing import Optional, Tuple

try:
    from numba import jit, prange, config as numba_config
    NUMBA_AVAILABLE = True
    # Enable Numba optimizations
    numba_config.THREADING_LAYER = 'omp'
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. Install numba for performance acceleration.")

# =====================================================
# Core Linear Algebra Kernels
# =====================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_matmul_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        2D matrix multiplication: C = A @ B
        Fully parallelized with simple iteration.
        
        Args:
            a: (M, K) matrix
            b: (K, N) matrix
            
        Returns:
            c: (M, N) result matrix
        """
        m, k = a.shape
        n = b.shape[1]
        c = np.zeros((m, n), dtype=a.dtype)
        
        # Parallelize over rows
        for i in prange(m):
            for j in range(n):
                sum_val = 0.0
                for kk in range(k):
                    sum_val += a[i, kk] * b[kk, j]
                c[i, j] = sum_val
        
        return c

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_matmul_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        3D batched matrix multiplication (batch matmul): C = A @ B
        Useful for transformer attention, batch processing.
        
        Args:
            a: (batch, m, k)
            b: (batch, k, n) or (k, n) for broadcast
            
        Returns:
            c: (batch, m, n)
        """
        if b.ndim == 2:
            # Broadcast case: b is (k, n)
            batch_size, m, k = a.shape
            n = b.shape[1]
            c = np.zeros((batch_size, m, n), dtype=a.dtype)
            
            for batch in prange(batch_size):
                for i in range(m):
                    for j in range(n):
                        sum_val = 0.0
                        for kk in range(k):
                            sum_val += a[batch, i, kk] * b[kk, j]
                        c[batch, i, j] = sum_val
        else:
            # Standard batched case: b is (batch, k, n)
            batch_size, m, k = a.shape
            n = b.shape[2]
            c = np.zeros((batch_size, m, n), dtype=a.dtype)
            
            for batch in prange(batch_size):
                for i in range(m):
                    for j in range(n):
                        sum_val = 0.0
                        for kk in range(k):
                            sum_val += a[batch, i, kk] * b[batch, kk, j]
                        c[batch, i, j] = sum_val
        
        return c

else:
    def numba_matmul_2d(a, b):
        """Fallback: NumPy matmul"""
        return np.dot(a, b)

    def numba_matmul_3d(a, b):
        """Fallback: NumPy batched matmul"""
        return np.matmul(a, b)


# =====================================================
# Convolution Kernels
# =====================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_conv2d_valid(input_array: np.ndarray, kernel_array: np.ndarray) -> np.ndarray:
        """
        2D convolution with 'valid' padding (no padding).
        Output size: (H - KH + 1) x (W - KW + 1)
        
        Args:
            input_array: (batch, in_channels, height, width)
            kernel_array: (out_channels, in_channels, kernel_h, kernel_w)
            
        Returns:
            output_array: (batch, out_channels, out_height, out_width)
        """
        if input_array.ndim == 2:
            input_array = input_array.reshape(1, 1, input_array.shape[0], input_array.shape[1])
        elif input_array.ndim == 3:
            input_array = input_array.reshape(input_array.shape[0], 1, input_array.shape[1], input_array.shape[2])
        
        batch_size, in_channels, in_h, in_w = input_array.shape
        out_channels, _, kernel_h, kernel_w = kernel_array.shape
        out_h = in_h - kernel_h + 1
        out_w = in_w - kernel_w + 1
        
        output = np.zeros((batch_size, out_channels, out_h, out_w), dtype=input_array.dtype)
        
        for b in prange(batch_size):
            for oc in range(out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        sum_val = 0.0
                        for ic in range(in_channels):
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    sum_val += (input_array[b, ic, oh + kh, ow + kw] * 
                                              kernel_array[oc, ic, kh, kw])
                        output[b, oc, oh, ow] = sum_val
        
        return output

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_conv2d_same(input_array: np.ndarray, kernel_array: np.ndarray) -> np.ndarray:
        """
        2D convolution with 'same' padding.
        Output size matches input height/width.
        
        Args:
            input_array: (batch, in_channels, height, width)
            kernel_array: (out_channels, in_channels, kernel_h, kernel_w)
            
        Returns:
            output_array: (batch, out_channels, height, width)
        """
        batch_size, in_channels, in_h, in_w = input_array.shape
        out_channels, _, kernel_h, kernel_w = kernel_array.shape
        
        # Calculate padding
        pad_h = kernel_h // 2
        pad_w = kernel_w // 2
        
        output = np.zeros((batch_size, out_channels, in_h, in_w), dtype=input_array.dtype)
        
        for b in prange(batch_size):
            for oc in range(out_channels):
                for oh in range(in_h):
                    for ow in range(in_w):
                        sum_val = 0.0
                        for ic in range(in_channels):
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    in_h_idx = oh + kh - pad_h
                                    in_w_idx = ow + kw - pad_w
                                    if 0 <= in_h_idx < in_h and 0 <= in_w_idx < in_w:
                                        sum_val += (input_array[b, ic, in_h_idx, in_w_idx] * 
                                                  kernel_array[oc, ic, kh, kw])
                        output[b, oc, oh, ow] = sum_val
        
        return output

else:
    def numba_conv2d_valid(input_array, kernel_array):
        """Fallback: Simple loop-based convolution"""
        batch, in_ch, in_h, in_w = input_array.shape
        out_ch, _, kh, kw = kernel_array.shape
        out_h, out_w = in_h - kh + 1, in_w - kw + 1
        output = np.zeros((batch, out_ch, out_h, out_w), dtype=input_array.dtype)
        for b in range(batch):
            for oc in range(out_ch):
                for oh in range(out_h):
                    for ow in range(out_w):
                        for ic in range(in_ch):
                            output[b, oc, oh, ow] += np.sum(
                                input_array[b, ic, oh:oh+kh, ow:ow+kw] * 
                                kernel_array[oc, ic, :, :]
                            )
        return output

    def numba_conv2d_same(input_array, kernel_array):
        """Fallback: NumPy-based convolution with padding"""
        from scipy import signal
        batch, in_ch, in_h, in_w = input_array.shape
        out_ch, _, kh, kw = kernel_array.shape
        output = np.zeros((batch, out_ch, in_h, in_w), dtype=input_array.dtype)
        for b in range(batch):
            for oc in range(out_ch):
                for ic in range(in_ch):
                    output[b, oc] += signal.correlate2d(
                        input_array[b, ic], kernel_array[oc, ic], mode='same'
                    )
        return output


# =====================================================
# Activation Functions
# =====================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, x)"""
        y = np.empty_like(x)
        flat_x = x.ravel()
        flat_y = y.ravel()
        
        for i in prange(flat_x.size):
            flat_y[i] = max(0.0, flat_x[i])
        
        return y.reshape(x.shape)

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Softmax activation with numerical stability.
        Simplified version that handles last axis only for Numba compatibility.
        """
        # For simplicity in nopython mode, we handle the common case (axis=-1)
        if x.ndim == 1:
            # 1D input
            output = np.empty_like(x)
            max_val = x[0]
            for i in range(1, len(x)):
                if x[i] > max_val:
                    max_val = x[i]
            
            exp_sum = 0.0
            for i in range(len(x)):
                output[i] = np.exp(x[i] - max_val)
                exp_sum += output[i]
            
            for i in range(len(x)):
                output[i] /= exp_sum
            
            return output
        
        elif x.ndim == 2:
            # 2D input - apply softmax along last dimension (axis=-1)
            batch, features = x.shape
            output = np.empty_like(x)
            
            for b in prange(batch):
                max_val = x[b, 0]
                for f in range(1, features):
                    if x[b, f] > max_val:
                        max_val = x[b, f]
                
                exp_sum = 0.0
                for f in range(features):
                    output[b, f] = np.exp(x[b, f] - max_val)
                    exp_sum += output[b, f]
                
                for f in range(features):
                    output[b, f] /= exp_sum
            
            return output
        
        else:
            # 3D+ input - flatten and reshape
            original_shape = x.shape
            x_2d = x.reshape(-1, x.shape[-1])
            output = np.empty_like(x_2d)
            batch = x_2d.shape[0]
            features = x_2d.shape[1]
            
            for b in prange(batch):
                max_val = x_2d[b, 0]
                for f in range(1, features):
                    if x_2d[b, f] > max_val:
                        max_val = x_2d[b, f]
                
                exp_sum = 0.0
                for f in range(features):
                    output[b, f] = np.exp(x_2d[b, f] - max_val)
                    exp_sum += output[b, f]
                
                for f in range(features):
                    output[b, f] /= exp_sum
            
            return output.reshape(original_shape)

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_gelu(x: np.ndarray) -> np.ndarray:
        """GELU activation (Gaussian Error Linear Unit)"""
        y = np.empty_like(x)
        flat_x = x.ravel()
        flat_y = y.ravel()
        cdf_const = 0.044715  # sqrt(2/pi)
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        
        for i in prange(flat_x.size):
            x_val = flat_x[i]
            # GELU(x) = x * Phi(x) where Phi is CDF of standard normal
            # Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            tanh_arg = sqrt_2_over_pi * (x_val + cdf_const * x_val * x_val * x_val)
            flat_y[i] = 0.5 * x_val * (1.0 + np.tanh(tanh_arg))
        
        return y

else:
    def numba_relu(x):
        """Fallback: NumPy ReLU"""
        return np.maximum(0.0, x)

    def numba_softmax(x, axis=-1):
        """Fallback: NumPy softmax"""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def numba_gelu(x):
        """Fallback: NumPy GELU approximation"""
        cdf_const = 0.044715
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + cdf_const * x**3)))


# =====================================================
# Normalization Kernels
# =====================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
                        epsilon: float = 1e-5) -> np.ndarray:
        """
        Layer normalization: applies to last dimension.
        
        Args:
            x: (batch, ..., features)
            gamma: (features,) scale
            beta: (features,) shift
            epsilon: numerical stability constant
            
        Returns:
            normalized output
        """
        shape = x.shape
        features = shape[-1]
        x_reshaped = x.reshape(-1, features)
        output = np.empty_like(x_reshaped)
        
        for i in prange(x_reshaped.shape[0]):
            # Compute mean and variance
            mean = 0.0
            for j in range(features):
                mean += x_reshaped[i, j]
            mean /= features
            
            variance = 0.0
            for j in range(features):
                diff = x_reshaped[i, j] - mean
                variance += diff * diff
            variance /= features
            
            # Normalize and apply scale/shift
            std = np.sqrt(variance + epsilon)
            for j in range(features):
                normalized = (x_reshaped[i, j] - mean) / std
                output[i, j] = normalized * gamma[j] + beta[j]
        
        return output.reshape(shape)

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_batch_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                        running_mean: np.ndarray, running_var: np.ndarray,
                        momentum: float = 0.9, epsilon: float = 1e-5,
                        training: bool = True) -> np.ndarray:
        """
        Batch normalization for 4D tensors (NCHW format).
        
        Args:
            x: (batch, channels, height, width)
            gamma, beta: (channels,)
            running_mean, running_var: (channels,) - updated during training
            
        Returns:
            normalized tensor
        """
        batch, channels, height, width = x.shape
        x_flat = x.reshape(batch, channels, -1)
        output = np.empty_like(x)
        output_flat = output.reshape(batch, channels, -1)
        
        for c in prange(channels):
            if training:
                # Compute batch statistics
                mean = 0.0
                count = 0
                for b in range(batch):
                    for hw in range(x_flat.shape[2]):
                        mean += x_flat[b, c, hw]
                        count += 1
                mean /= count
                
                variance = 0.0
                for b in range(batch):
                    for hw in range(x_flat.shape[2]):
                        diff = x_flat[b, c, hw] - mean
                        variance += diff * diff
                variance /= count
                
                # Update running statistics
                running_mean[c] = momentum * running_mean[c] + (1.0 - momentum) * mean
                running_var[c] = momentum * running_var[c] + (1.0 - momentum) * variance
            else:
                # Use running statistics
                mean = running_mean[c]
                variance = running_var[c]
            
            # Normalize
            std = np.sqrt(variance + epsilon)
            for b in range(batch):
                for hw in range(x_flat.shape[2]):
                    normalized = (x_flat[b, c, hw] - mean) / std
                    output_flat[b, c, hw] = normalized * gamma[c] + beta[c]
        
        return output

else:
    def numba_layer_norm(x, gamma, beta, epsilon=1e-5):
        """Fallback: NumPy layer norm"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + epsilon) * gamma + beta

    def numba_batch_norm(x, gamma, beta, running_mean, running_var, 
                        momentum=0.9, epsilon=1e-5, training=True):
        """Fallback: Simple batch norm"""
        if training:
            axes = tuple(range(x.ndim - 1))
            mean = np.mean(x, axis=axes, keepdims=True)
            var = np.var(x, axis=axes, keepdims=True)
        else:
            shape = [1] * x.ndim
            shape[1] = gamma.shape[0]
            mean = running_mean.reshape(shape)
            var = running_var.reshape(shape)
        return (x - mean) / np.sqrt(var + epsilon) * gamma.reshape(shape) + beta.reshape(shape)


# =====================================================
# Element-wise Operations
# =====================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise addition"""
        result = np.empty_like(a)
        flat_a = a.ravel()
        flat_b = b.ravel()
        flat_r = result.ravel()
        
        for i in prange(flat_a.size):
            flat_r[i] = flat_a[i] + flat_b[i]
        
        return result

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise multiplication"""
        result = np.empty_like(a)
        flat_a = a.ravel()
        flat_b = b.ravel()
        flat_r = result.ravel()
        
        for i in prange(flat_a.size):
            flat_r[i] = flat_a[i] * flat_b[i]
        
        return result

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_scale(a: np.ndarray, scalar: float) -> np.ndarray:
        """Scalar multiplication"""
        result = np.empty_like(a)
        flat_a = a.ravel()
        flat_r = result.ravel()
        
        for i in prange(flat_a.size):
            flat_r[i] = flat_a[i] * scalar
        
        return result

else:
    def numba_add(a, b):
        return a + b
    
    def numba_mul(a, b):
        return a * b
    
    def numba_scale(a, scalar):
        return a * scalar


# =====================================================
# Pooling Operations
# =====================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_max_pool2d(x: np.ndarray, pool_size: int, stride: int) -> np.ndarray:
        """
        Max pooling for 4D input.
        
        Args:
            x: (batch, channels, height, width)
            pool_size: size of pooling window
            stride: stride of pooling
            
        Returns:
            pooled output
        """
        batch, channels, height, width = x.shape
        out_h = (height - pool_size) // stride + 1
        out_w = (width - pool_size) // stride + 1
        
        output = np.empty((batch, channels, out_h, out_w), dtype=x.dtype)
        
        for b in prange(batch):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * stride
                        w_start = ow * stride
                        h_end = h_start + pool_size
                        w_end = w_start + pool_size
                        
                        max_val = x[b, c, h_start, w_start]
                        for h in range(h_start, h_end):
                            for w in range(w_start, w_end):
                                if x[b, c, h, w] > max_val:
                                    max_val = x[b, c, h, w]
                        
                        output[b, c, oh, ow] = max_val
        
        return output

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_avg_pool2d(x: np.ndarray, pool_size: int, stride: int) -> np.ndarray:
        """Average pooling for 4D input."""
        batch, channels, height, width = x.shape
        out_h = (height - pool_size) // stride + 1
        out_w = (width - pool_size) // stride + 1
        
        output = np.empty((batch, channels, out_h, out_w), dtype=x.dtype)
        pool_area = pool_size * pool_size
        
        for b in prange(batch):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * stride
                        w_start = ow * stride
                        h_end = h_start + pool_size
                        w_end = w_start + pool_size
                        
                        sum_val = 0.0
                        for h in range(h_start, h_end):
                            for w in range(w_start, w_end):
                                sum_val += x[b, c, h, w]
                        
                        output[b, c, oh, ow] = sum_val / pool_area
        
        return output

else:
    def numba_max_pool2d(x, pool_size, stride):
        """Fallback: loop-based max pooling"""
        batch, channels, height, width = x.shape
        out_h = (height - pool_size) // stride + 1
        out_w = (width - pool_size) // stride + 1
        output = np.zeros((batch, channels, out_h, out_w), dtype=x.dtype)
        for b in range(batch):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start, w_start = oh * stride, ow * stride
                        output[b, c, oh, ow] = np.max(
                            x[b, c, h_start:h_start+pool_size, w_start:w_start+pool_size]
                        )
        return output

    def numba_avg_pool2d(x, pool_size, stride):
        """Fallback: loop-based average pooling"""
        batch, channels, height, width = x.shape
        out_h = (height - pool_size) // stride + 1
        out_w = (width - pool_size) // stride + 1
        output = np.zeros((batch, channels, out_h, out_w), dtype=x.dtype)
        for b in range(batch):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start, w_start = oh * stride, ow * stride
                        output[b, c, oh, ow] = np.mean(
                            x[b, c, h_start:h_start+pool_size, w_start:w_start+pool_size]
                        )
        return output


# =====================================================
# Reduction Operations
# =====================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_reduce_sum(x: np.ndarray, axis: Optional[int] = None, 
                        keepdims: bool = False) -> np.ndarray:
        """Parallel sum reduction"""
        if axis is None:
            return np.array([np.sum(x)])
        
        # Handle negative axis
        if axis < 0:
            axis = x.ndim + axis
        
        out_shape = list(x.shape)
        out_shape.pop(axis)
        if keepdims:
            out_shape.insert(axis, 1)
        
        output = np.zeros(out_shape, dtype=x.dtype)
        
        # This is a simplified version; full axis handling would be more complex
        if x.ndim == 2 and axis == 0:
            for j in prange(x.shape[1]):
                sum_val = 0.0
                for i in range(x.shape[0]):
                    sum_val += x[i, j]
                output[j] = sum_val
        elif x.ndim == 2 and axis == 1:
            for i in prange(x.shape[0]):
                sum_val = 0.0
                for j in range(x.shape[1]):
                    sum_val += x[i, j]
                output[i] = sum_val
        else:
            output = np.sum(x, axis=axis, keepdims=keepdims)
        
        return output

else:
    def numba_reduce_sum(x, axis=None, keepdims=False):
        """Fallback: NumPy sum"""
        return np.sum(x, axis=axis, keepdims=keepdims)


# =====================================================
# Public API
# =====================================================

def get_numba_status() -> dict:
    """Returns status of Numba availability and configuration"""
    status = {
        'numba_available': NUMBA_AVAILABLE,
        'threading_enabled': False,
        'platform': 'CPU'
    }
    
    if NUMBA_AVAILABLE:
        status['threading_enabled'] = True
        try:
            import numba
            status['numba_version'] = numba.__version__
        except:
            status['numba_version'] = 'unknown'
    
    return status


# =====================================================
# Backward Pass Kernels (Ultra-Optimized)
# =====================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_relu_backward(grad_output: np.ndarray, input_data: np.ndarray) -> np.ndarray:
        """
        Backward pass for ReLU activation.
        Optimized: single-pass computation with mask.
        
        Returns:
            grad_input: gradient w.r.t. input
        """
        grad_input = np.empty_like(grad_output)
        flat_grad = grad_output.ravel()
        flat_input = input_data.ravel()
        flat_result = grad_input.ravel()
        
        for i in prange(flat_grad.size):
            flat_result[i] = flat_grad[i] if flat_input[i] > 0.0 else 0.0
        
        return grad_input.reshape(grad_output.shape)

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_matmul_2d_backward(grad_output: np.ndarray, input_a: np.ndarray,
                                input_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass for 2D matrix multiplication.
        Optimized: parallel computation of both gradients.
        
        dL/dA = dL/dC @ B^T
        dL/dB = A^T @ dL/dC
        """
        m, k = input_a.shape
        n = grad_output.shape[1]
        
        # Gradient w.r.t. A: (grad_output @ B^T)
        grad_a = np.zeros((m, k), dtype=grad_output.dtype)
        for i in prange(m):
            for kk in range(k):
                sum_val = 0.0
                for j in range(n):
                    sum_val += grad_output[i, j] * input_b[j, kk]
                grad_a[i, kk] = sum_val
        
        # Gradient w.r.t. B: (A^T @ grad_output)
        grad_b = np.zeros((k, n), dtype=grad_output.dtype)
        for kk in prange(k):
            for j in range(n):
                sum_val = 0.0
                for i in range(m):
                    sum_val += input_a[i, kk] * grad_output[i, j]
                grad_b[kk, j] = sum_val
        
        return grad_a, grad_b

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_conv2d_backward_kernel(grad_output: np.ndarray, input_data: np.ndarray,
                                     kernel: np.ndarray, stride: int = 1) -> np.ndarray:
        """
        Backward pass for 2D convolution (kernel gradients only).
        Optimized for minimal allocations.
        
        Returns:
            grad_kernel: gradient w.r.t. kernel
        """
        batch_size, in_channels, in_h, in_w = input_data.shape
        out_channels, _, kh, kw = kernel.shape
        batch_size_out, out_channels_out, out_h, out_w = grad_output.shape
        
        grad_kernel = np.zeros_like(kernel)
        
        for b in prange(batch_size):
            for oc in range(out_channels):
                for ic in range(in_channels):
                    for h in range(out_h):
                        for w in range(out_w):
                            h_start = h * stride
                            w_start = w * stride
                            h_end = h_start + kh
                            w_end = w_start + kw
                            
                            for kh_idx in range(kh):
                                for kw_idx in range(kw):
                                    grad_kernel[oc, ic, kh_idx, kw_idx] += (
                                        input_data[b, ic, h_start + kh_idx, w_start + kw_idx] *
                                        grad_output[b, oc, h, w]
                                    )
        
        return grad_kernel

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_softmax_backward(grad_output: np.ndarray, softmax_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for softmax activation.
        Optimized: vectorized jacobian computation.
        
        Jacobian: diag(s) - s @ s^T
        Result: (diag(s) - s @ s^T) @ grad = s * grad - s * (s @ grad)
        """
        batch_size, num_classes = grad_output.shape
        grad_input = np.zeros_like(grad_output)
        
        for b in prange(batch_size):
            # Compute s @ grad (dot product)
            s_dot_g = 0.0
            for i in range(num_classes):
                s_dot_g += softmax_output[b, i] * grad_output[b, i]
            
            # Compute result: s * grad - s * (s @ grad)
            for i in range(num_classes):
                grad_input[b, i] = (
                    softmax_output[b, i] * grad_output[b, i] -
                    softmax_output[b, i] * s_dot_g
                )
        
        return grad_input

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_add_backward(grad_output: np.ndarray, _unused: np.ndarray) -> np.ndarray:
        """
        Backward pass for addition operation.
        Gradient simply passes through.
        """
        return grad_output.copy()

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def numba_bias_backward(grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for bias addition.
        Optimized: parallel reduction along batch dimension.
        """
        batch_size = grad_output.shape[0]
        num_features = grad_output.shape[1] if grad_output.ndim > 1 else 1
        
        if grad_output.ndim == 2:
            grad_bias = np.zeros(num_features, dtype=grad_output.dtype)
            for j in prange(num_features):
                sum_val = 0.0
                for i in range(batch_size):
                    sum_val += grad_output[i, j]
                grad_bias[j] = sum_val
        else:
            grad_bias = np.sum(grad_output)
        
        return grad_bias

else:
    # Fallback implementations (for when Numba is not available)
    pass


__all__ = [
    'NUMBA_AVAILABLE',
    'numba_matmul_2d',
    'numba_matmul_3d',
    'numba_conv2d_valid',
    'numba_conv2d_same',
    'numba_relu',
    'numba_softmax',
    'numba_gelu',
    'numba_layer_norm',
    'numba_batch_norm',
    'numba_add',
    'numba_mul',
    'numba_scale',
    'numba_max_pool2d',
    'numba_avg_pool2d',
    'numba_reduce_sum',
    'numba_relu_backward',
    'numba_matmul_2d_backward',
    'numba_conv2d_backward_kernel',
    'numba_softmax_backward',
    'numba_add_backward',
    'numba_bias_backward',
    'get_numba_status',
]
