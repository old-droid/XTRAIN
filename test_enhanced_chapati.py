#!/usr/bin/env python3

"""
Test script for Enhanced Chapati LM with Novel Formulas and Retry Architecture
"""

import sys
import os
import numpy as np

# Add XTRAIN to Python path
xtrain_path = os.path.join(os.path.dirname(__file__), 'XTRAIN')
if xtrain_path not in sys.path:
    sys.path.insert(0, xtrain_path)

from chapati_core import ChapatiLM, TekkenTokenizer, SimpleEnglishDataset, ChapatiLMTrainer, generate_sample_text

def test_enhanced_architecture():
    """Test the enhanced Chapati LM architecture"""
    print("Testing Enhanced Chapati LM Architecture...")
    
    # Test 1: Initialize enhanced tokenizer
    print("\n1. Testing Enhanced Tokenizer...")
    tokenizer = TekkenTokenizer(vocab_size=5000)
    print(f"   Tokenizer initialized with {tokenizer.get_vocab_size()} tokens")
    
    # Test 2: Initialize enhanced model with retry mechanism
    print("\n2. Testing Enhanced Chapati LM with Retry Mechanism...")
    model = ChapatiLM(
        vocab_size=tokenizer.get_vocab_size(), 
        d_model=256,  # Reduced for faster testing
        num_workers=2, 
        num_thoughts=2,
        max_retries=1, 
        retry_threshold=0.2
    )
    print(f"   Model initialized with retry mechanism (max_retries={model.max_retries})")
    
    # Test 3: Test novel formulas
    print("\n3. Testing Novel Mathematical Formulas...")
    
    # Create test input with valid token IDs (integers within vocab range)
    test_input = np.random.randint(0, model.vocab_size, (1, 5)).astype(np.int32)  # batch_size=1, seq_len=5
    
    # Test forward pass with retry mechanism
    try:
        output_logits = model.forward(test_input)
        print(f"   Forward pass successful! Output shape: {output_logits.shape}")
        
        # Test novel confusion score calculation - use correct dimension
        test_hidden = np.random.randn(model.d_model).astype(np.float32)  # Use model's d_model
        confusion_score = model._calculate_confusion_score(test_hidden)
        print(f"   Novel confusion score: {confusion_score:.4f}")
        
        # Test novel confidence score calculation
        test_logits = np.random.randn(model.vocab_size).astype(np.float32)  # Use model's vocab size
        confidence_score = model._calculate_confidence_score(test_logits)
        print(f"   Novel confidence score: {confidence_score:.4f}")
        
        # Test adaptive retry decision
        should_retry = model._adaptive_retry_decision(confidence_score, 0)
        print(f"   Adaptive retry decision: {should_retry}")
        
    except Exception as e:
        print(f"   Error in novel formulas: {e}")
        import traceback
        traceback.print_exc()
        
        # Debug step by step
        print("\n   Debugging step by step:")
        try:
            print("   - Testing forward pass...")
            print(f"     Test input shape: {test_input.shape}")
            print(f"     Test input values: {test_input}")
            print(f"     Embedding layer shape: {model.embedding_layer.shape}")
            output_logits = model.forward(test_input)
            print(f"     Forward pass OK: {output_logits.shape}")
        except Exception as e1:
            print(f"     Forward pass failed: {e1}")
        
        try:
            print("   - Testing confusion score...")
            test_hidden = np.random.randn(model.d_model).astype(np.float32)
            print(f"     Hidden state shape: {test_hidden.shape}")
            confusion_score = model._calculate_confusion_score(test_hidden)
            print(f"     Confusion score OK: {confusion_score:.4f}")
        except Exception as e2:
            print(f"     Confusion score failed: {e2}")
        
        try:
            print("   - Testing confidence score...")
            test_logits = np.random.randn(model.vocab_size).astype(np.float32)
            confidence_score = model._calculate_confidence_score(test_logits)
            print(f"     Confidence score OK: {confidence_score:.4f}")
        except Exception as e3:
            print(f"     Confidence score failed: {e3}")
        
        return False
    
    # Test 4: Test enhanced tokenizer
    print("\n4. Testing Enhanced Tokenizer...")
    test_text = "Enhanced Chapati LM with novel formulas and retry architecture"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"   Original: {test_text}")
    print(f"   Encoded length: {len(encoded)}")
    print(f"   Decoded: {decoded}")
    print(f"   Round-trip successful: {test_text == decoded}")
    
    # Test 5: Test performance metrics
    print("\n5. Testing Enhanced Performance Metrics...")
    metrics = model.get_performance_metrics()
    print(f"   Total tokens processed: {metrics['total_tokens']}")
    print(f"   Worker hits: {metrics['worker_hits']}")
    print(f"   Thought engine hits: {metrics['thought_engine_hits']}")
    print(f"   Retry attempts: {metrics['retry_attempts']}")
    print(f"   Retry successes: {metrics['retry_successes']}")
    print(f"   Combined efficiency: {metrics['combined_efficiency']:.3f}")
    
    # Test 6: Quick training test (small dataset)
    print("\n6. Testing Training with Enhanced Architecture...")
    try:
        # Create small dataset
        small_dataset = SimpleEnglishDataset()
        
        # Train for 1 epoch with small batch
        trainer = ChapatiLMTrainer(model, tokenizer, learning_rate=0.001)
        
        # Use only first 4 samples for quick test
        print("   Training on small subset (4 samples, 1 epoch)...")
        
        # Manual training loop for quick test
        tokenized_samples = [tokenizer.encode(sample) for sample in small_dataset.get_samples()[:4]]
        max_len = max(len(sample) for sample in tokenized_samples)
        
        input_data = []
        for sample in tokenized_samples:
            if len(sample) < max_len:
                padded_sample = sample + [tokenizer.special_tokens['<pad>']] * (max_len - len(sample))
            else:
                padded_sample = sample[:max_len]
            input_data.append(padded_sample)
        
        input_data = np.array(input_data, dtype=np.int32)
        
        # Create targets (shifted inputs)
        target_data = np.zeros_like(input_data)
        target_data[:, :-1] = input_data[:, 1:]
        target_data[:, -1] = tokenizer.special_tokens['<eos>']
        
        # Single training step
        loss = trainer.train_step(input_data, target_data)
        print(f"   Training step completed! Loss: {loss:.4f}")
        
    except Exception as e:
        print(f"   Training test error: {e}")
        return False
    
    # Test 7: Test generation
    print("\n7. Testing Text Generation...")
    try:
        sample_text = generate_sample_text(
            model, tokenizer, 
            prompt="Enhanced AI", 
            length=10, 
            temperature=0.8, 
            top_k=20
        )
        print(f"   Generated text: {sample_text}")
    except Exception as e:
        print(f"   Generation error: {e}")
        return False
    
    print("\n[SUCCESS] All tests passed! Enhanced Chapati LM architecture is working correctly.")
    return True

if __name__ == "__main__":
    success = test_enhanced_architecture()
    if success:
        print("\n[SUCCESS] Enhanced Chapati LM with Novel Formulas and Retry Architecture is fully functional!")
    else:
        print("\n[ERROR] Some tests failed. Please check the implementation.")
