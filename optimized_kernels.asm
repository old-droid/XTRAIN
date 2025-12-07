; ============================================================================
; CPUWARP-ML: AMD64 Optimized Kernels (AVX2 + FMA) - Windows MSVC Format
; ============================================================================
; Syntax: MASM (Microsoft Macro Assembler)
; Calling Convention: Microsoft x64 (rcx, rdx, r8, r9 for first 4 args)
; Assemble: ml64 /c optimized_kernels_amd64.asm /Fo optimized_kernels_amd64.obj
; Link: link /dll optimized_kernels_amd64.obj /out:optimized_kernels_amd. dll

. code

; ============================================================================
; optimized_matmul: Matrix Multiplication (MxK @ KxN -> MxN)
; void optimized_matmul(float* A, float* B, float* C, int M, int K, int N)
; Arguments: rcx=A, rdx=B, r8=C, r9d=M, [rsp+40]=K, [rsp+48]=N
; ============================================================================

optimized_matmul PROC
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    ; Arguments
    mov r10, rcx            ; A pointer
    mov r11, rdx            ; B pointer
    mov r12, r8             ; C pointer
    mov r13d, r9d           ; M
    mov r14d, [rbp + 16]    ; K (at rbp+16 after push)
    mov r15d, [rbp + 24]    ; N (at rbp+24 after push)
    
    ; Initialize loop counter
    xor ebx, ebx            ; i = 0
    
. loop_i:
    cmp ebx, r13d           ; if i >= M
    jge .done_i
    
    xor eax, eax            ; j = 0
    
.loop_j:
    cmp eax, r15d           ; if j >= N
    jge .done_j
    
    ; Initialize accumulator for 8 floats (ymm0)
    vxorps ymm0, ymm0, ymm0
    
    xor ecx, ecx            ; k = 0
    
.loop_k:
    cmp ecx, r14d           ; if k >= K
    jge .store_result
    
    ; Calculate A[i*K + k]
    mov r8, rbx
    imul r8, r14            ; r8 = i * K
    add r8, rcx             ; r8 = i*K + k
    shl r8, 2               ; r8 = (i*K+k) * 4
    
    ; Load A[i*K + k] and broadcast
    vbroadcastss ymm1, dword ptr [r10 + r8]
    
    ; Calculate B[k*N + j]
    mov r9, rcx
    imul r9, r15            ; r9 = k * N
    add r9, rax             ; r9 = k*N + j
    shl r9, 2               ; r9 = (k*N+j) * 4
    
    ; Load 8 floats from B[k*N + j]
    vmovups ymm2, ymmword ptr [r11 + r9]
    
    ; FMA: ymm0 += ymm1 * ymm2
    vfmadd231ps ymm0, ymm1, ymm2
    
    inc ecx                 ; k++
    jmp .loop_k
    
.store_result:
    ; Calculate C[i*N + j]
    mov r8, rbx
    imul r8, r15            ; r8 = i * N
    add r8, rax             ; r8 = i*N + j
    shl r8, 2               ; r8 = (i*N+j) * 4
    
    ; Store 8 floats to C[i*N + j]
    vmovups ymmword ptr [r12 + r8], ymm0
    
    add eax, 8              ; j += 8
    jmp .loop_j
    
.done_j:
    inc ebx                 ; i++
    jmp .loop_i
    
.done_i:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
optimized_matmul ENDP

; ============================================================================
; optimized_relu: ReLU Activation (max(x, 0))
; void optimized_relu(float* input, float* output, int size)
; Arguments: rcx=input, rdx=output, r8d=size
; ============================================================================

optimized_relu PROC
    push rbp
    mov rbp, rsp
    
    vxorps ymm1, ymm1, ymm1    ; Zero vector
    xor rax, rax               ; i = 0
    
.relu_loop:
    cmp rax, r8                ; if i >= size
    jge .relu_done
    
    ; Load 8 floats from input
    vmovups ymm0, ymmword ptr [rcx + rax*4]
    
    ; ReLU: max(ymm0, 0)
    vmaxps ymm0, ymm0, ymm1
    
    ; Store result
    vmovups ymmword ptr [rdx + rax*4], ymm0
    
    add rax, 8                 ; i += 8
    jmp .relu_loop
    
.relu_done:
    vzeroupper
    pop rbp
    ret
optimized_relu ENDP

; ============================================================================
; optimized_softmax: Softmax Activation
; void optimized_softmax(float* input, float* output, int size)
; Arguments: rcx=input, rdx=output, r8d=size
; ============================================================================

optimized_softmax PROC
    push rbp
    mov rbp, rsp
    push rbx
    
    ; Find maximum value
    movss xmm0, dword ptr [rcx]  ; max = input[0]
    mov rax, 1
    
.find_max_loop:
    cmp rax, r8
    jge .max_found
    
    movss xmm1, dword ptr [rcx + rax*4]
    maxss xmm0, xmm1
    inc rax
    jmp .find_max_loop
    
. max_found:
    vbroadcastss ymm1, xmm0     ; Broadcast max
    
    ; Compute exp(x - max) and sum
    xor rax, rax
    vxorps ymm2, ymm2, ymm2     ; sum = 0
    
.exp_loop:
    cmp rax, r8
    jge .exp_done
    
    vmovups ymm0, ymmword ptr [rcx + rax*4]
    vsubps ymm0, ymm0, ymm1     ; x - max
    
    ; Store for now (exp would need approximation or library call)
    vmovups ymmword ptr [rdx + rax*4], ymm0
    vaddps ymm2, ymm2, ymm0     ; sum += (x - max)
    
    add rax, 8
    jmp .exp_loop
    
.exp_done:
    vzeroupper
    pop rbx
    pop rbp
    ret
optimized_softmax ENDP

; ============================================================================
; optimized_add: Element-wise Addition
; void optimized_add(float* a, float* b, float* c, int size)
; Arguments: rcx=a, rdx=b, r8=c, r9d=size
; ============================================================================

optimized_add PROC
    push rbp
    mov rbp, rsp
    
    xor rax, rax
    
.add_loop:
    cmp rax, r9              ; if i >= size
    jge .add_done
    
    vmovups ymm0, ymmword ptr [rcx + rax*4]
    vmovups ymm1, ymmword ptr [rdx + rax*4]
    vaddps ymm0, ymm0, ymm1
    vmovups ymmword ptr [r8 + rax*4], ymm0
    
    add rax, 8
    jmp .add_loop
    
.add_done:
    vzeroupper
    pop rbp
    ret
optimized_add ENDP

; ============================================================================
; optimized_mul: Element-wise Multiplication
; void optimized_mul(float* a, float* b, float* c, int size)
; Arguments: rcx=a, rdx=b, r8=c, r9d=size
; ============================================================================

optimized_mul PROC
    push rbp
    mov rbp, rsp
    
    xor rax, rax
    
.mul_loop:
    cmp rax, r9
    jge .mul_done
    
    vmovups ymm0, ymmword ptr [rcx + rax*4]
    vmovups ymm1, ymmword ptr [rdx + rax*4]
    vmulps ymm0, ymm0, ymm1
    vmovups ymmword ptr [r8 + rax*4], ymm0
    
    add rax, 8
    jmp .mul_loop
    
.mul_done:
    vzeroupper
    pop rbp
    ret
optimized_mul ENDP

END