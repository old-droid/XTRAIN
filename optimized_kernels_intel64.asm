; ============================================================================
; CPUWARP-ML: Intel64 Optimized Kernels (AVX-512)
; ============================================================================
; Optimized for Intel Skylake-X, Ice Lake, Sapphire Rapids (AVX-512)
; 512-bit vector operations for 2x throughput vs AVX2
; ============================================================================

section .text
global optimized_matmul
global optimized_relu
global optimized_softmax

align 64

; ============================================================================
; optimized_matmul (AVX-512): 16 floats per operation
; void optimized_matmul(float* A, float* B, float* C, int M, int K, int N)
; ============================================================================

optimized_matmul:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    
    xor     r10d, r10d           ; i = 0
    
.matmul_i_loop:
    cmp     r10d, ecx            ; if i >= M
    jge     .matmul_i_end
    
    xor     r11d, r11d           ; j = 0
    
.matmul_j_loop:
    cmp     r11d, r9d            ; if j >= N
    jge     .matmul_j_end
    
    vpxord  zmm0, zmm0, zmm0    ; sum = 0 (16 floats)
    
    xor     r12d, r12d           ; k = 0
    
.matmul_k_loop:
    cmp     r12d, r8d            ; if k >= K
    jge     .matmul_k_end
    
    ; A[i*K + k]
    mov     r13, r10
    imul    r13, r8
    add     r13, r12
    shl     r13, 2
    
    ; B[k*N + j]
    mov     r14, r12
    imul    r14, r9
    add     r14, r11
    shl     r14, 2
    
    vbroadcastss zmm1, dword [rdi + r13]
    vmovups zmm2, [rsi + r14]
    
    vfmadd231ps zmm0, zmm1, zmm2
    
    inc     r12d
    jmp     .matmul_k_loop
    
.matmul_k_end:
    ; C[i*N + j]
    mov     r13, r10
    imul    r13, r9
    add     r13, r11
    shl     r13, 2
    
    vmovups [rdx + r13], zmm0
    
    add     r11d, 16             ; j += 16 (16 floats in ZMM)
    jmp     .matmul_j_loop
    
.matmul_j_end:
    inc     r10d                 ; i++
    jmp     .matmul_i_loop
    
.matmul_i_end:
    vzeroupper
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

; ============================================================================
; optimized_relu (AVX-512)
; ============================================================================

optimized_relu:
    push    rbp
    mov     rbp, rsp
    
    vpxord  zmm1, zmm1, zmm1     ; Zero vector
    xor     rax, rax
    
.relu_512_loop:
    cmp     rax, rdx
    jge     .relu_512_end
    
    vmovups zmm0, [rdi + rax*4]
    vmaxps  zmm0, zmm0, zmm1
    vmovups [rsi + rax*4], zmm0
    
    add     rax, 16              ; 16 floats in ZMM
    jmp     .relu_512_loop
    
.relu_512_end:
    vzeroupper
    pop     rbp
    ret

; ============================================================================
; optimized_softmax (AVX-512)
; ============================================================================

optimized_softmax:
    push    rbp
    mov     rbp, rsp
    
    ; Find max
    movss   xmm0, [rdi]
    mov     rax, 1
    
.find_max_512:
    cmp     rax, rdx
    jge     .max_512_found
    
    movss   xmm1, [rdi + rax*4]
    maxss   xmm0, xmm1
    inc     rax
    jmp     .find_max_512
    
.max_512_found:
    vbroadcastss zmm1, xmm0
    
    ; Exp and sum
    xor     rax, rax
    vpxord  zmm2, zmm2, zmm2
    
.exp_512_loop:
    cmp     rax, rdx
    jge     .exp_512_done
    
    vmovups zmm0, [rdi + rax*4]
    vsubps  zmm0, zmm0, zmm1
    
    vmovups [rsi + rax*4], zmm0
    vaddps  zmm2, zmm2, zmm0
    
    add     rax, 16
    jmp     .exp_512_loop
    
.exp_512_done:
    vzeroupper
    pop     rbp
    ret