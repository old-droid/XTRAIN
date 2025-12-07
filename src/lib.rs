use std::arch::x86_64::*;

static mut HAS_AVX2: bool = false;
static mut HAS_FMA: bool = false;
static mut HAS_AVX512F: bool = false;

#[no_mangle]
pub unsafe extern "C" fn detect_cpu_features() {
    HAS_AVX2 = is_x86_feature_detected!("avx2");
    HAS_FMA = is_x86_feature_detected!("fma");
    HAS_AVX512F = is_x86_feature_detected!("avx512f");
}

#[no_mangle]
pub unsafe extern "C" fn optimized_matmul(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i32,
    k: i32,
    n: i32,
) {
    detect_cpu_features([package]
name = "xtrain"
version = "0.1.0"
edition = "2021"

[dependencies]
rand = "0.8"

[build-dependencies]
cc = "1.0"
nasm = "0.2"
);

    const BLOCK_SIZE_M: i32 = 64;
    const BLOCK_SIZE_K: i32 = 256;
    const BLOCK_SIZE_N: i32 = 64;

    for bi in (0..m).step_by(BLOCK_SIZE_M as usize) {
        for bj in (0..n).step_by(BLOCK_SIZE_N as usize) {
            for bk in (0..k).step_by(BLOCK_SIZE_K as usize) {
                let end_i = (bi + BLOCK_SIZE_M).min(m);
                let end_j = (bj + BLOCK_SIZE_N).min(n);
                let end_k = (bk + BLOCK_SIZE_K).min(k);

                for i in bi..end_i {
                    for j in (bj..end_j).step_by(8) {
                        let mut sum = _mm256_setzero_ps();

                        for kk in bk..end_k {
                            let a_vec = _mm256_broadcast_ss(a.offset((i * k + kk) as isize));
                            let b_vec = _mm256_loadu_ps(b.offset((kk * n + j) as isize));

                            if HAS_FMA {
                                sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
                            } else {
                                sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
                            }
                        }

                        if bk == 0 {
                            _mm256_storeu_ps(c.offset((i * n + j) as isize), sum);
                        } else {
                            let c_vec = _mm256_loadu_ps(c.offset((i * n + j) as isize));
                            _mm256_storeu_ps(c.offset((i * n + j) as isize), _mm256_add_ps(c_vec, sum));
                        }
                    }
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn optimized_conv2d(
    input: *const f32,
    kernel: *const f32,
    output: *mut f32,
    batch_size: i32,
    in_channels: i32,
    in_height: i32,
    in_width: i32,
    out_channels: i32,
    kernel_height: i32,
    kernel_width: i32,
) {
    detect_cpu_features();

    let out_height = in_height - kernel_height + 1;
    let out_width = in_width - kernel_width + 1;

    for b in 0..batch_size {
        for oc in 0..out_channels {
            for oh in 0..out_height {
                for ow in (0..out_width).step_by(8) {
                    let mut sum = _mm256_setzero_ps();

                    for ic in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let in_h = oh + kh;
                                let in_w = ow + kw;

                                let kernel_val = *kernel.offset(
                                    (((oc * in_channels + ic) * kernel_height + kh) * kernel_width
                                        + kw) as isize,
                                );
                                let kernel_vec = _mm256_broadcast_ss(&kernel_val);

                                let mut input_vals = [0.0f32; 8];
                                for i in 0..8 {
                                    if ow + i < out_width {
                                        let idx = ((b * in_channels + ic) * in_height + in_h)
                                            * in_width
                                            + in_w
                                            + i;
                                        if in_w + i < in_width {
                                            input_vals[i as usize] = *input.offset(idx as isize);
                                        }
                                    }
                                }
                                let input_vec = _mm256_loadu_ps(input_vals.as_ptr());

                                if HAS_FMA {
                                    sum = _mm256_fmadd_ps(kernel_vec, input_vec, sum);
                                } else {
                                    sum = _mm256_add_ps(sum, _mm256_mul_ps(kernel_vec, input_vec));
                                }
                            }
                        }
                    }

                    let mut result_vals = [0.0f32; 8];
                    _mm256_storeu_ps(result_vals.as_mut_ptr(), sum);

                    for i in 0..8 {
                        if ow + i < out_width {
                            let out_idx = ((b * out_channels + oc) * out_height + oh) * out_width
                                + ow
                                + i;
                            *output.offset(out_idx as isize) = result_vals[i as usize];
                        }
                    }
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn optimized_relu(input: *const f32, output: *mut f32, size: i32) {
    detect_cpu_features();

    let zero = _mm256_setzero_ps();
    for i in (0..size).step_by(8) {
        let x = _mm256_loadu_ps(input.offset(i as isize));
        let result = _mm256_max_ps(x, zero);
        _mm256_storeu_ps(output.offset(i as isize), result);
    }

    for i in (size & !7)..size {
        *output.offset(i as isize) = (*input.offset(i as isize)).max(0.0);
    }
}

#[no_mangle]
pub unsafe extern "C" fn optimized_softmax(input: *const f32, output: *mut f32, size: i32) {
    detect_cpu_features();

    let mut max_val = *input;
    for i in 1..size {
        if *input.offset(i as isize) > max_val {
            max_val = *input.offset(i as isize);
        }
    }

    let max_vec = _mm256_broadcast_ss(&max_val);

    let mut sum = 0.0;
    for i in (0..size).step_by(8) {
        let x = _mm256_loadu_ps(input.offset(i as isize));
        let exp_x = _mm256_exp_ps(_mm256_sub_ps(x, max_vec));
        _mm256_storeu_ps(output.offset(i as isize), exp_x);

        let mut exp_vals = [0.0f32; 8];
        _mm256_storeu_ps(exp_vals.as_mut_ptr(), exp_x);
        for j in 0..8 {
            if i + j < size {
                sum += exp_vals[j as usize];
            }
        }
    }

    let sum_vec = _mm256_broadcast_ss(&sum);
    for i in (0..size).step_by(8) {
        let exp_x = _mm256_loadu_ps(output.offset(i as isize));
        let result = _mm256_div_ps(exp_x, sum_vec);
        _mm256_storeu_ps(output.offset(i as isize), result);
    }
}

#[no_mangle]
pub unsafe extern "C" fn optimized_add(a: *const f32, b: *const f32, c: *mut f32, size: i32) {
    for i in (0..size).step_by(8) {
        let va = _mm256_loadu_ps(a.offset(i as isize));
        let vb = _mm256_loadu_ps(b.offset(i as isize));
        let vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c.offset(i as isize), vc);
    }

    for i in (size & !7)..size {
        *c.offset(i as isize) = *a.offset(i as isize) + *b.offset(i as isize);
    }
}

#[no_mangle]
pub unsafe extern "C" fn optimized_mul(a: *const f32, b: *const f32, c: *mut f32, size: i32) {
    for i in (0..size).step_by(8) {
        let va = _mm256_loadu_ps(a.offset(i as isize));
        let vb = _mm256_loadu_ps(b.offset(i as isize));
        let vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(c.offset(i as isize), vc);
    }

    for i in (size & !7)..size {
        *c.offset(i as isize) = *a.offset(i as isize) * *b.offset(i as isize);
    }
}

#[no_mangle]
pub unsafe extern "C" fn optimized_layer_norm(
    input: *const f32,
    output: *mut f32,
    gamma: *const f32,
    beta: *const f32,
    batch_size: i32,
    features: i32,
    epsilon: f32,
) {
    for b in 0..batch_size {
        let x = input.offset((b * features) as isize);
        let y = output.offset((b * features) as isize);

        let mut sum_vec = _mm256_setzero_ps();
        for i in (0..features).step_by(8) {
            let x_vec = _mm256_loadu_ps(x.offset(i as isize));
            sum_vec = _mm256_add_ps(sum_vec, x_vec);
        }

        let mut sum_vals = [0.0f32; 8];
        _mm256_storeu_ps(sum_vals.as_mut_ptr(), sum_vec);
        let mut mean = 0.0;
        for i in 0..8 {
            mean += sum_vals[i];
        }
        for i in (features & !7)..features {
            mean += *x.offset(i as isize);
        }
        mean /= features as f32;

        let mean_vec = _mm256_broadcast_ss(&mean);
        let mut var_sum = _mm256_setzero_ps();
        for i in (0..features).step_by(8) {
            let x_vec = _mm256_loadu_ps(x.offset(i as isize));
            let diff = _mm256_sub_ps(x_vec, mean_vec);
            var_sum = _mm256_fmadd_ps(diff, diff, var_sum);
        }

        _mm256_storeu_ps(sum_vals.as_mut_ptr(), var_sum);
        let mut variance = 0.0;
        for i in 0..8 {
            variance += sum_vals[i];
        }
        for i in (features & !7)..features {
            let diff = *x.offset(i as isize) - mean;
            variance += diff * diff;
        }
        variance /= features as f32;

        let inv_std = 1.0 / (variance + epsilon).sqrt();
        let inv_std_vec = _mm256_broadcast_ss(&inv_std);

        for i in (0..features).step_by(8) {
            let x_vec = _mm256_loadu_ps(x.offset(i as isize));
            let gamma_vec = _mm256_loadu_ps(gamma.offset(i as isize));
            let beta_vec = _mm256_loadu_ps(beta.offset(i as isize));

            let norm = _mm256_mul_ps(_mm256_sub_ps(x_vec, mean_vec), inv_std_vec);
            let result = _mm256_fmadd_ps(norm, gamma_vec, beta_vec);

            _mm256_storeu_ps(y.offset(i as isize), result);
        }

        for i in (features & !7)..features {
            let norm = (*x.offset(i as isize) - mean) * inv_std;
            *y.offset(i as isize) = norm * *gamma.offset(i as isize) + *beta.offset(i as isize);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn get_cpu_features() -> i32 {
    detect_cpu_features();
    ((HAS_AVX512F as i32) << 2) | ((HAS_FMA as i32) << 1) | (HAS_AVX2 as i32)
}

#[no_mangle]
pub unsafe extern "C" fn aligned_malloc(size: usize, alignment: usize) -> *mut f32 {
    let layout = std::alloc::Layout::from_size_align(size, alignment).unwrap();
    std::alloc::alloc(layout) as *mut f32
}

#[no_mangle]
pub unsafe extern "C" fn aligned_free(ptr: *mut f32, size: usize, alignment: usize) {
    let layout = std::alloc::Layout::from_size_align(size, alignment).unwrap();
    std::alloc::dealloc(ptr as *mut u8, layout);
}

#[no_mangle]
pub unsafe extern "C" fn benchmark_matmul(m: i32, k: i32, n: i32, iterations: i32) -> f64 {
    let a_size = (m * k) as usize * std::mem::size_of::<f32>();
    let b_size = (k * n) as usize * std::mem::size_of::<f32>();
    let c_size = (m * n) as usize * std::mem::size_of::<f32>();

    let a = aligned_malloc(a_size, 32);
    let b = aligned_malloc(b_size, 32);
    let c = aligned_malloc(c_size, 32);

    for i in 0..(m * k) {
        *a.offset(i as isize) = rand::random::<f32>();
    }
    for i in 0..(k * n) {
        *b.offset(i as isize) = rand::random::<f32>();
    }

    let start = std::time::Instant::now();

    for _ in 0..iterations {
        optimized_matmul(a, b, c, m, k, n);
    }

    let end = std::time::Instant::now();

    aligned_free(a, a_size, 32);
    aligned_free(b, b_size, 32);
    aligned_free(c, c_size, 32);

    (end - start).as_secs_f64() / iterations as f64
}

// Added the following to the code to make it work
#[no_mangle]
pub extern "C" fn __cpuid_count(
    leaf: ::std::os::raw::c_uint,
    subleaf: ::std::os::raw::c_uint,
    eax: *mut ::std::os::raw::c_uint,
    ebx: *mut ::std::os::raw::c_uint,
    ecx: *mut ::std::os::raw::c_uint,
    edx: *mut ::std::os::raw::c_uint,
) {
    unsafe {
        asm!(
            "cpuid",
            inout("eax") leaf => *eax,
            inout("ebx") 0 => *ebx,
            inout("ecx") subleaf => *ecx,
            inout("edx") 0 => *edx,
        );
    }
}
#[no_mangle]
pub extern "C" fn __cpuid(
    leaf: ::std::os::raw::c_uint,
    eax: *mut ::std::os::raw::c_uint,
    ebx: *mut ::std::os::raw::c_uint,
    ecx: *mut ::std::os::raw::c_uint,
    edx: *mut ::std::os::raw::c_uint,
) {
    unsafe {
        asm!(
            "cpuid",
            inout("eax") leaf => *eax,
            inout("ebx") 0 => *ebx,
            inout("ecx") 0 => *ecx,
            inout("edx") 0 => *edx,
        );
    }
}
#[no_mangle]
pub unsafe extern "C" fn _mm256_exp_ps(a: __m256) -> __m256 {
    let mut dst = _mm256_setzero_ps();
    let mut src = [0.0; 8];
    _mm256_storeu_ps(src.as_mut_ptr(), a);
    for i in 0..8 {
        src[i] = src[i].exp();
    }
    dst = _mm256_loadu_ps(src.as_ptr());
    dst
}
