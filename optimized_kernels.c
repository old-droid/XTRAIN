/*
 * CPUWARP-ML Optimized C Kernels
 * High-performance SIMD-optimized kernels for AMD and Intel CPUs
 * Supports AVX2, FMA, and AVX-512 instructions
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

// CPU feature detection
static int has_avx2 = -1;
static int has_fma = -1;
static int has_avx512f = -1;

void detect_cpu_features() {
    if (has_avx2 == -1) {
        unsigned int eax, ebx, ecx, edx;
        
        // Check for AVX2 (CPUID.07H:EBX.AVX2[bit 5])
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        has_avx2 = (ebx & (1 << 5)) != 0;
        has_avx512f = (ebx & (1 << 16)) != 0;
        
        // Check for FMA (CPUID.01H:ECX.FMA[bit 12])
        __cpuid(1, eax, ebx, ecx, edx);
        has_fma = (ecx & (1 << 12)) != 0;
    }
}

// Optimized matrix multiplication with cache blocking and SIMD
EXPORT void optimized_matmul(float* A, float* B, float* C, int M, int K, int N) {
    detect_cpu_features();
    
    // Cache-friendly block sizes optimized for L1/L2 cache
    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_K = 256;
    const int BLOCK_SIZE_N = 64;
    
    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < M; bi += BLOCK_SIZE_M) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE_N) {
            for (int bk = 0; bk < K; bk += BLOCK_SIZE_K) {
                
                int end_i = (bi + BLOCK_SIZE_M < M) ? bi + BLOCK_SIZE_M : M;
                int end_j = (bj + BLOCK_SIZE_N < N) ? bj + BLOCK_SIZE_N : N;
                int end_k = (bk + BLOCK_SIZE_K < K) ? bk + BLOCK_SIZE_K : K;
                
                // Inner kernel with SIMD optimization
                for (int i = bi; i < end_i; i++) {
                    for (int j = bj; j < end_j; j += 8) {
                        __m256 sum = _mm256_setzero_ps();
                        
                        for (int k = bk; k < end_k; k++) {
                            __m256 a_vec = _mm256_broadcast_ss(&A[i * K + k]);
                            __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                            
                            if (has_fma) {
                                sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
                            } else {
                                sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
                            }
                        }
                        
                        if (bk == 0) {
                            _mm256_storeu_ps(&C[i * N + j], sum);
                        } else {
                            __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                            _mm256_storeu_ps(&C[i * N + j], _mm256_add_ps(c_vec, sum));
                        }
                    }
                }
            }
        }
    }
}

// Optimized 2D convolution with im2col transformation
EXPORT void optimized_conv2d(float* input, float* kernel, float* output,
                           int batch_size, int in_channels, int in_height, int in_width,
                           int out_channels, int kernel_height, int kernel_width) {
    
    detect_cpu_features();
    
    int out_height = in_height - kernel_height + 1;
    int out_width = in_width - kernel_width + 1;
    
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow += 8) {
                    
                    __m256 sum = _mm256_setzero_ps();
                    
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                            for (int kw = 0; kw < kernel_width; kw++) {
                                
                                int in_h = oh + kh;
                                int in_w = ow + kw;
                                
                                float kernel_val = kernel[((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw];
                                __m256 kernel_vec = _mm256_broadcast_ss(&kernel_val);
                                
                                // Load input values (handle boundary)
                                float input_vals[8];
                                for (int i = 0; i < 8 && (ow + i) < out_width; i++) {
                                    int idx = ((b * in_channels + ic) * in_height + in_h) * in_width + in_w + i;
                                    input_vals[i] = (in_w + i < in_width) ? input[idx] : 0.0f;
                                }
                                __m256 input_vec = _mm256_loadu_ps(input_vals);
                                
                                if (has_fma) {
                                    sum = _mm256_fmadd_ps(kernel_vec, input_vec, sum);
                                } else {
                                    sum = _mm256_add_ps(sum, _mm256_mul_ps(kernel_vec, input_vec));
                                }
                            }
                        }
                    }
                    
                    // Store results (handle boundary)
                    float result_vals[8];
                    _mm256_storeu_ps(result_vals, sum);
                    
                    for (int i = 0; i < 8 && (ow + i) < out_width; i++) {
                        int out_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow + i;
                        output[out_idx] = result_vals[i];
                    }
                }
            }
        }
    }
}

// Optimized ReLU activation
EXPORT void optimized_relu(float* input, float* output, int size) {
    detect_cpu_features();
    
    __m256 zero = _mm256_setzero_ps();
    
    #pragma omp parallel for
    for (int i = 0; i < size; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 result = _mm256_max_ps(x, zero);
        _mm256_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (int i = (size & ~7); i < size; i++) {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

// Optimized softmax activation
EXPORT void optimized_softmax(float* input, float* output, int size) {
    detect_cpu_features();
    
    // Find maximum for numerical stability
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    __m256 max_vec = _mm256_broadcast_ss(&max_val);
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 exp_x = _mm256_exp_ps(_mm256_sub_ps(x, max_vec));
        _mm256_storeu_ps(&output[i], exp_x);
        
        // Sum for normalization
        float exp_vals[8];
        _mm256_storeu_ps(exp_vals, exp_x);
        for (int j = 0; j < 8 && (i + j) < size; j++) {
            sum += exp_vals[j];
        }
    }
    
    // Normalize by sum
    __m256 sum_vec = _mm256_broadcast_ss(&sum);
    
    #pragma omp parallel for
    for (int i = 0; i < size; i += 8) {
        __m256 exp_x = _mm256_loadu_ps(&output[i]);
        __m256 result = _mm256_div_ps(exp_x, sum_vec);
        _mm256_storeu_ps(&output[i], result);
    }
}

// Vectorized element-wise operations
EXPORT void optimized_add(float* a, float* b, float* c, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&c[i], vc);
    }
    
    // Handle remaining elements
    for (int i = (size & ~7); i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

EXPORT void optimized_mul(float* a, float* b, float* c, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&c[i], vc);
    }
    
    // Handle remaining elements
    for (int i = (size & ~7); i < size; i++) {
        c[i] = a[i] * b[i];
    }
}

// Layer normalization
EXPORT void optimized_layer_norm(float* input, float* output, float* gamma, float* beta,
                                int batch_size, int features, float epsilon) {
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        float* x = &input[b * features];
        float* y = &output[b * features];
        
        // Compute mean
        __m256 sum_vec = _mm256_setzero_ps();
        for (int i = 0; i < features; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(&x[i]);
            sum_vec = _mm256_add_ps(sum_vec, x_vec);
        }
        
        float sum_vals[8];
        _mm256_storeu_ps(sum_vals, sum_vec);
        float mean = 0.0f;
        for (int i = 0; i < 8; i++) mean += sum_vals[i];
        for (int i = (features & ~7); i < features; i++) mean += x[i];
        mean /= features;
        
        // Compute variance
        __m256 mean_vec = _mm256_broadcast_ss(&mean);
        __m256 var_sum = _mm256_setzero_ps();
        
        for (int i = 0; i < features; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(&x[i]);
            __m256 diff = _mm256_sub_ps(x_vec, mean_vec);
            var_sum = _mm256_fmadd_ps(diff, diff, var_sum);
        }
        
        _mm256_storeu_ps(sum_vals, var_sum);
        float variance = 0.0f;
        for (int i = 0; i < 8; i++) variance += sum_vals[i];
        for (int i = (features & ~7); i < features; i++) {
            float diff = x[i] - mean;
            variance += diff * diff;
        }
        variance /= features;
        
        // Normalize and scale
        float inv_std = 1.0f / sqrtf(variance + epsilon);
        __m256 inv_std_vec = _mm256_broadcast_ss(&inv_std);
        
        for (int i = 0; i < features; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(&x[i]);
            __m256 gamma_vec = _mm256_loadu_ps(&gamma[i]);
            __m256 beta_vec = _mm256_loadu_ps(&beta[i]);
            
            __m256 norm = _mm256_mul_ps(_mm256_sub_ps(x_vec, mean_vec), inv_std_vec);
            __m256 result = _mm256_fmadd_ps(norm, gamma_vec, beta_vec);
            
            _mm256_storeu_ps(&y[i], result);
        }
        
        // Handle remaining elements
        for (int i = (features & ~7); i < features; i++) {
            float norm = (x[i] - mean) * inv_std;
            y[i] = norm * gamma[i] + beta[i];
        }
    }
}

// Get CPU feature information
EXPORT int get_cpu_features() {
    detect_cpu_features();
    return (has_avx512f << 2) | (has_fma << 1) | has_avx2;
}

// Memory alignment utilities
EXPORT float* aligned_malloc(size_t size, size_t alignment) {
    #ifdef _WIN32
        return (float*)_aligned_malloc(size, alignment);
    #else
        void* ptr;
        if (posix_memalign(&ptr, alignment, size) != 0) {
            return NULL;
        }
        return (float*)ptr;
    #endif
}

EXPORT void aligned_free(float* ptr) {
    #ifdef _WIN32
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

// Benchmark utilities
EXPORT double benchmark_matmul(int M, int K, int N, int iterations) {
    float* A = aligned_malloc(M * K * sizeof(float), 32);
    float* B = aligned_malloc(K * N * sizeof(float), 32);
    float* C = aligned_malloc(M * N * sizeof(float), 32);
    
    // Initialize with random data
    for (int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX;
    
    double start = omp_get_wtime();
    
    for (int iter = 0; iter < iterations; iter++) {
        optimized_matmul(A, B, C, M, K, N);
    }
    
    double end = omp_get_wtime();
    
    aligned_free(A);
    aligned_free(B);
    aligned_free(C);
    
    return (end - start) / iterations;
}