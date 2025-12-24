#!/usr/bin/env python3
"""
Test Neural Orchestration Integration
======================================

Comprehensive test suite for the neural orchestration system integrated into Chapati LM.
Tests all components: worker nodes, orchestrator, manager node, safety guardrail, 
verifier block, and bounded retry policy.
"""

import sys
import os
import numpy as np

# Add XTRAIN to Python path
xtrain_path = os.path.join(os.path.dirname(__file__), 'XTRAIN')
if xtrain_path not in sys.path:
    sys.path.insert(0, xtrain_path)

from chapati_core import NeuralOrchestrationSystem, ChapatiLM, TekkenTokenizer

def test_neural_orchestration_system():
    """Test NeuralOrchestrationSystem independently"""
    print("Testing NeuralOrchestrationSystem...")
    
    # Create neural orchestration system
    orchestration = NeuralOrchestrationSystem(
        num_workers=3,
        num_neurons=6,
        max_retries=2,
        d_model=256
    )
    
    # Test input
    batch_size = 4
    test_input = np.random.randn(batch_size, 256).astype(np.float32)
    
    # Forward pass
    try:
        outputs, orchestration_info = orchestration.forward(test_input)
        print("[SUCCESS] Neural orchestration forward pass successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Orchestration info keys: {list(orchestration_info.keys())}")
        
        # Verify output dimensions
        assert outputs.shape == (batch_size, 256), f"Expected output shape (4, 256), got {outputs.shape}"
        
        # Verify orchestration info contains expected fields
        expected_fields = ['neuron_scores', 'composite_scores', 'safety_scores', 
                          'verifier_scores', 'acceptance_decisions', 'retry_count', 'routing_decisions']
        for field in expected_fields:
            assert field in orchestration_info, f"Missing field: {field}"
        
        print("[SUCCESS] All orchestration outputs and metadata verified")
        
    except Exception as e:
        print(f"[FAILED] Neural orchestration test failed: {e}")
        return False
    
    # Test metrics
    try:
        metrics = orchestration.get_orchestration_metrics()
        print(f"[SUCCESS] Orchestration metrics retrieved: {len(metrics)} metrics")
        
        # Verify key metrics are present
        expected_metrics = ['worker_outputs', 'orchestrator_scores', 'manager_routing_decisions',
                           'safety_filter_activations', 'verifier_acceptances', 'verifier_rejections',
                           'retry_attempts', 'retry_successes', 'unsafe_content_blocked']
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        print("[SUCCESS] All orchestration metrics verified")
        
    except Exception as e:
        print(f"[FAILED] Metrics test failed: {e}")
        return False
    
    return True

def test_integrated_chapati_lm():
    """Test ChapatiLM with integrated neural orchestration"""
    print("\nTesting ChapatiLM with Neural Orchestration...")
    
    # Create tokenizer and model
    try:
        tokenizer = TekkenTokenizer(vocab_size=10000)
        model = ChapatiLM(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=256,
            num_workers=3,
            num_thoughts=2,
            max_retries=2,
            retry_threshold=0.4,
            num_neurons=6
        )
        print("[SUCCESS] ChapatiLM with neural orchestration initialized")
        
    except Exception as e:
        print(f"[FAILED] Model initialization failed: {e}")
        return False
    
    # Test forward pass
    try:
        # Create test input
        test_text = "Hello world this is a test"
        input_ids = tokenizer.encode(test_text)
        input_array = np.array([input_ids])  # Add batch dimension
        
        print(f"[SUCCESS] Test input prepared: '{test_text}'")
        print(f"  Input IDs shape: {input_array.shape}")
        
        # Forward pass
        output_logits = model.forward(input_array)
        
        print("[SUCCESS] Forward pass successful")
        print(f"  Output logits shape: {output_logits.shape}")
        
        # Verify output dimensions
        expected_shape = (1, len(input_ids), tokenizer.get_vocab_size())
        assert output_logits.shape == expected_shape, f"Expected {expected_shape}, got {output_logits.shape}"
        
        print("[SUCCESS] Output dimensions verified")
        
    except Exception as e:
        print(f"[FAILED] Forward pass test failed: {e}")
        return False
    
    # Test performance metrics with orchestration
    try:
        metrics = model.get_performance_metrics()
        print(f"[SUCCESS] Performance metrics retrieved: {len(metrics)} metrics")
        
        # Verify orchestration metrics are included
        assert 'orchestration_metrics' in metrics, "Missing orchestration metrics"
        orchestration_metrics = metrics['orchestration_metrics']
        
        # Verify key orchestration metrics
        expected_orchestration_metrics = ['worker_outputs', 'orchestrator_scores', 
                                         'manager_routing_decisions', 'safety_filter_activations']
        
        for metric in expected_orchestration_metrics:
            assert metric in orchestration_metrics, f"Missing orchestration metric: {metric}"
        
        print("[SUCCESS] Orchestration metrics in performance analysis verified")
        
        # Check for new efficiency metrics
        expected_efficiency_metrics = ['orchestration_efficiency', 'safety_effectiveness', 
                                      'verifier_acceptance_rate', 'orchestration_optimization']
        
        for metric in expected_efficiency_metrics:
            assert metric in metrics, f"Missing efficiency metric: {metric}"
        
        print("[SUCCESS] Enhanced efficiency metrics verified")
        
    except Exception as e:
        print(f"[FAILED] Performance metrics test failed: {e}")
        return False
    
    return True

def test_retry_mechanism_with_orchestration():
    """Test the enhanced retry mechanism with neural orchestration"""
    print("\nTesting Enhanced Retry Mechanism with Neural Orchestration...")
    
    try:
        tokenizer = TekkenTokenizer(vocab_size=5000)
        model = ChapatiLM(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=128,
            num_workers=2,
            num_thoughts=1,
            max_retries=3,  # Higher retries for testing
            retry_threshold=0.2,  # Lower threshold to trigger retries
            num_neurons=4
        )
        
        print("[SUCCESS] Model with enhanced retry mechanism initialized")
        
        # Create challenging input that might trigger retries
        challenging_text = "This is a complex and ambiguous sentence that might cause confusion in the model"
        input_ids = tokenizer.encode(challenging_text)
        input_array = np.array([input_ids])
        
        print(f"[SUCCESS] Challenging input prepared: '{challenging_text[:50]}...'")
        
        # Forward pass
        output_logits = model.forward(input_array)
        
        # Get metrics
        metrics = model.get_performance_metrics()
        
        print("[SUCCESS] Forward pass with challenging input completed")
        print(f"  Retry attempts: {metrics['retry_attempts']}")
        print(f"  Retry successes: {metrics['retry_successes']}")
        print(f"  Retry success rate: {metrics['retry_success_rate']:.3f}")
        
        # Verify retry metrics are reasonable
        assert metrics['retry_attempts'] >= 0, "Negative retry attempts"
        assert metrics['retry_successes'] >= 0, "Negative retry successes"
        
        if metrics['retry_attempts'] > 0:
            assert metrics['retry_success_rate'] >= 0.0, "Invalid retry success rate"
            assert metrics['retry_success_rate'] <= 1.0, "Invalid retry success rate"
            print("[SUCCESS] Retry mechanism working with neural orchestration guidance")
        else:
            print("[SUCCESS] No retries needed - input processed successfully on first attempt")
        
    except Exception as e:
        print(f"[FAILED] Retry mechanism test failed: {e}")
        return False
    
    return True

def test_safety_guardrail():
    """Test the safety guardrail functionality"""
    print("\nTesting Safety Guardrail Functionality...")
    
    try:
        # Create orchestration system
        orchestration = NeuralOrchestrationSystem(
            num_workers=2,
            num_neurons=4,
            max_retries=1,
            d_model=128
        )
        
        # Test with potentially unsafe content
        batch_size = 8
        test_input = np.random.randn(batch_size, 128).astype(np.float32)
        
        # Add some patterns that might trigger safety filtering
        # (in practice, this would be handled by the bad matrices)
        test_input[:, :10] = 5.0  # Exaggerated values that might be flagged
        
        # Forward pass
        outputs, orchestration_info = orchestration.forward(test_input)
        
        # Check safety metrics
        metrics = orchestration.get_orchestration_metrics()
        
        print("[SUCCESS] Safety guardrail test completed")
        print(f"  Safety filter activations: {metrics['safety_filter_activations']}")
        print(f"  Unsafe content blocked: {metrics['unsafe_content_blocked']}")
        print(f"  Safety scores range: {np.min(orchestration_info['safety_scores']):.3f} - {np.max(orchestration_info['safety_scores']):.3f}")
        
        # Verify safety metrics are non-negative
        assert metrics['safety_filter_activations'] >= 0, "Negative safety activations"
        assert metrics['unsafe_content_blocked'] >= 0, "Negative unsafe content count"
        
        print("[SUCCESS] Safety guardrail metrics verified")
        
    except Exception as e:
        print(f"[FAILED] Safety guardrail test failed: {e}")
        return False
    
    return True

def test_end_to_end_integration():
    """End-to-end test of the complete integrated system"""
    print("\nTesting End-to-End Integration...")
    
    try:
        # Initialize components
        tokenizer = TekkenTokenizer(vocab_size=8000)
        model = ChapatiLM(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=192,
            num_workers=3,
            num_thoughts=2,
            max_retries=2,
            retry_threshold=0.3,
            num_neurons=5
        )
        
        print("[SUCCESS] End-to-end system initialized")
        
        # Test multiple sentences
        test_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming industries worldwide.",
            "This is a complex sentence that tests the neural orchestration system."
        ]
        
        all_success = True
        for i, sentence in enumerate(test_sentences):
            try:
                print(f"\n  Processing sentence {i+1}: '{sentence[:40]}...'")
                
                # Tokenize and process
                input_ids = tokenizer.encode(sentence)
                input_array = np.array([input_ids])
                
                # Forward pass
                output_logits = model.forward(input_array)
                
                # Verify output
                expected_shape = (1, len(input_ids), tokenizer.get_vocab_size())
                assert output_logits.shape == expected_shape
                
                print(f"    [SUCCESS] Sentence {i+1} processed successfully")
                
            except Exception as e:
                print(f"    [FAILED] Sentence {i+1} failed: {e}")
                all_success = False
        
        if all_success:
            print("[SUCCESS] All sentences processed successfully")
            
            # Get final metrics
            metrics = model.get_performance_metrics()
            print(f"\n  Final Performance Summary:")
            print(f"    Total tokens processed: {metrics['total_tokens']}")
            print(f"    Worker hits: {metrics['worker_hits']}")
            print(f"    Thought engine hits: {metrics['thought_engine_hits']}")
            print(f"    Retry attempts: {metrics['retry_attempts']}")
            print(f"    Combined efficiency: {metrics['combined_efficiency']:.3f}")
            print(f"    Orchestration efficiency: {metrics['orchestration_efficiency']:.3f}")
            print(f"    Safety effectiveness: {metrics['safety_effectiveness']:.3f}")
            
        return all_success
        
    except Exception as e:
        print(f"[FAILED] End-to-end integration test failed: {e}")
        return False

def main():
    """Run all neural orchestration integration tests"""
    print("=" * 60)
    print("NEURAL ORCHESTRATION INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Neural Orchestration System", test_neural_orchestration_system),
        ("Integrated Chapati LM", test_integrated_chapati_lm),
        ("Enhanced Retry Mechanism", test_retry_mechanism_with_orchestration),
        ("Safety Guardrail", test_safety_guardrail),
        ("End-to-End Integration", test_end_to_end_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-' * 50}")
        print(f"Running: {test_name}")
        print(f"{'-' * 50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            status = "PASSED" if success else "FAILED"
            print(f"\n[SUCCESS] {test_name}: {status}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n[FAILED] {test_name}: FAILED with exception: {e}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED! Neural orchestration integration successful!")
        return True
    else:
        print(f"\n[FAILED] {total - passed} test(s) failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)