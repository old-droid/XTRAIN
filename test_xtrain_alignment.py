import numpy as np
from numba import njit
import time
import sys
import os

# 1. FRAMEWORK IMPORTS
try:
    from numba_kernels import numba_matmul_2d, NUMBA_AVAILABLE
    from nn_layers import Dense, ReGLU
    print("[SUCCESS] Framework imports completed.")
except ImportError as e:
    print(f"[ERROR] Framework imports failed: {e}")
    # Try adding current directory to path if it fails
    sys.path.append(os.getcwd())
    try:
        from numba_kernels import numba_matmul_2d, NUMBA_AVAILABLE
        from nn_layers import Dense, ReGLU
        print("[SUCCESS] Framework imports completed after path adjustment.")
    except ImportError as e2:
        print(f"[ERROR] Still failed: {e2}")
        exit(1)

# 2. JIT COMPILATION CHECK
@njit(fastmath=True)
def dummy_matmul(a, b):
    m, k = a.shape
    n = b.shape[1]
    c = np.zeros((m, n), dtype=a.dtype)
    for i in range(m):
        for j in range(n):
            sum_val = 0.0
            for kk in range(k):
                sum_val += a[i, kk] * b[kk, j]
            c[i, j] = sum_val
    return c

def verify_jit():
    print("\n--- JIT Compilation Check ---")
    a = np.random.randn(64, 64).astype(np.float32)
    b = np.random.randn(64, 64).astype(np.float32)

    print("Compiling dummy_matmul...")
    # Warm up (compilation happens here)
    _ = dummy_matmul(a, b)

    start = time.time()
    _ = dummy_matmul(a, b)
    end = time.time()
    print(f"[SUCCESS] JIT matmul executed in {end - start:.6f}s")

    if NUMBA_AVAILABLE:
        print("Verifying framework's numba_matmul_2d...")
        start = time.time()
        _ = numba_matmul_2d(a, b)
        end = time.time()
        print(f"[SUCCESS] Framework matmul executed in {end - start:.6f}s")

# 3. CONTEXT ALIGNMENT
def check_alignment():
    print("\n--- Context Alignment Check ---")
    # Exact expected dimensions identified from debug_chapati.py and test_enhanced_chapati.py
    D_MODEL = 256
    VOCAB_SIZE = 5000
    BATCH_SIZE = 1
    SEQ_LEN = 5

    print(f"Identified Dimensions: d_model={D_MODEL}, vocab_size={VOCAB_SIZE}")

    # Mock hidden state
    hidden_state = np.random.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).astype(np.float32)
    print(f"Hidden state shape: {hidden_state.shape}")

    try:
        # Dense layer verification (Linear projection to vocab)
        dense = Dense(D_MODEL, VOCAB_SIZE)
        hidden_2d = hidden_state.reshape(-1, D_MODEL)
        logits = dense.forward(hidden_2d)
        print(f"[SUCCESS] Dense layer forward pass successful. Logits shape: {logits.shape}")

        # ReGLU verification (common in Chapati variants)
        # Note: ReGLU typically expects input_dim = 2 * output_dim due to splitting
        reglu = ReGLU()
        reglu_input = np.random.randn(BATCH_SIZE, SEQ_LEN, D_MODEL * 2).astype(np.float32)
        reglu_output = reglu.forward(reglu_input)
        print(f"[SUCCESS] ReGLU forward pass successful. Output shape: {reglu_output.shape}")

    except Exception as e:
        print(f"[ERROR] Alignment check failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("XTRAIN Framework Alignment Diagnostic")
    print("=====================================")
    verify_jit()
    check_alignment()
    print("\n[COMPLETE] All diagnostic steps finished.")