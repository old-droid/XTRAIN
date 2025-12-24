#!/usr/bin/env python3

"""
Debug script to identify the exact issue in enhanced Chapati LM
"""

import sys
import os
import numpy as np
import traceback

# Add XTRAIN to Python path
xtrain_path = os.path.join(os.path.dirname(__file__), 'XTRAIN')
if xtrain_path not in sys.path:
    sys.path.insert(0, xtrain_path)

from chapati_core import ChapatiLM, TekkenTokenizer

def debug_novel_formulas():
    """Debug the novel formulas step by step"""
    print("Debugging Novel Formulas...")
    
    # Initialize minimal setup
    tokenizer = TekkenTokenizer(vocab_size=100)
    model = ChapatiLM(
        vocab_size=tokenizer.get_vocab_size(), 
        d_model=64,  # Very small for debugging
        num_workers=1, 
        num_thoughts=1,
        max_retries=1, 
        retry_threshold=0.2
    )
    
    print(f"Model initialized: vocab={model.vocab_size}, d_model={model.d_model}")
    print(f"Entropy weights shape: {model.orchestrator['entropy_weights'].shape}")
    
    # Test 1: Simple confusion score calculation
    print("\n1. Testing confusion score calculation...")
    try:
        test_hidden = np.random.randn(64).astype(np.float32)  # 1D vector
        print(f"   Input hidden state shape: {test_hidden.shape}")
        
        # Step by step execution
        print("   Step 1: Reshape if needed")
        if test_hidden.ndim == 1:
            test_hidden_2d = test_hidden.reshape(1, -1)
            print(f"   Reshaped to: {test_hidden_2d.shape}")
        else:
            test_hidden_2d = test_hidden
        
        print("   Step 2: Matrix multiplication")
        entropy_weights = model.orchestrator['entropy_weights']
        print(f"   Entropy weights shape: {entropy_weights.shape}")
        
        # Test matrix multiplication
        try:
            confusion_logits = np.matmul(test_hidden_2d, entropy_weights)
            print(f"   Matrix multiplication successful! Output shape: {confusion_logits.shape}")
        except Exception as e:
            print(f"   Matrix multiplication failed: {e}")
            print(f"   test_hidden_2d shape: {test_hidden_2d.shape}")
            print(f"   entropy_weights shape: {entropy_weights.shape}")
            return False
        
        print("   Step 3: Entropy calculation")
        confusion_entropy = model._calculate_entropy(confusion_logits)
        print(f"   Confusion entropy: {confusion_entropy:.4f}")
        
        print("   Step 4: Full confusion score")
        confusion_score = model._calculate_confusion_score(test_hidden)
        print(f"   Full confusion score: {confusion_score:.4f}")
        
    except Exception as e:
        print(f"   Error in confusion score: {e}")
        traceback.print_exc()
        return False
    
    # Test 2: Confidence score calculation
    print("\n2. Testing confidence score calculation...")
    try:
        test_logits = np.random.randn(50).astype(np.float32)  # Small vocab size
        confidence_score = model._calculate_confidence_score(test_logits)
        print(f"   Confidence score: {confidence_score:.4f}")
    except Exception as e:
        print(f"   Error in confidence score: {e}")
        traceback.print_exc()
        return False
    
    # Test 3: Retry decision
    print("\n3. Testing retry decision...")
    try:
        should_retry = model._adaptive_retry_decision(0.5, 0)
        print(f"   Retry decision (confidence=0.5, retry_count=0): {should_retry}")
    except Exception as e:
        print(f"   Error in retry decision: {e}")
        traceback.print_exc()
        return False
    
    print("\n[SUCCESS] All novel formulas working correctly!")
    return True

if __name__ == "__main__":
    debug_novel_formulas()
