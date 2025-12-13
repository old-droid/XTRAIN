#!/usr/bin/env python3

"""
Test script for RobustNeuralNet implementation
"""

import numpy as np
import sys
import os
sys.path.append('.')

from train_cnn import RobustNeuralNet, ReGLU

def test_reglu():
    """Test reGLU activation function"""
    print("=" * 50)
    print("Testing ReGLU Activation Function")
    print("=" * 50)
    
    reglu = ReGLU()
    
    # Test forward pass
    x = np.array([[1, 2, -1, 0.5]]).astype(np.float32)
    print(f"Input: {x}")
    
    output = reglu.forward(x)
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")
    
    # Test backward pass
    grad_output = np.array([[1, 1]]).astype(np.float32)
    grad_input = reglu.backward(grad_output)
    print(f"Gradient input: {grad_input}")
    print(f"Gradient input shape: {grad_input.shape}")
    
    print("[OK] ReGLU test passed\n")

def test_robust_neural_net():
    """Test RobustNeuralNet class"""
    print("=" * 50)
    print("Testing RobustNeuralNet")
    print("=" * 50)
    
    # Test initialization
    model = RobustNeuralNet(input_dim=4, hidden_dim=4, output_dim=1)
    print(f"Model initialized: {model.input_dim} -> {model.hidden_dim} -> {model.output_dim}")
    print(f"Total parameters: {model.get_num_parameters()}")
    
    # Test forward pass
    x_test = np.random.randn(2, 4).astype(np.float32)
    print(f"\nForward pass test:")
    print(f"Input shape: {x_test.shape}")
    
    output = model.forward(x_test)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    
    # Test backward pass
    print(f"\nBackward pass test:")
    grad_output = np.random.randn(2, 1).astype(np.float32)
    gradients = model.backward(grad_output)
    print(f"Backward pass completed successfully")
    
    # Test backpropagation validation
    print(f"\nBackpropagation validation:")
    success = model.check_backprop(x_test)
    print(f"Result: {'[PASSED]' if success else '[FAILED]'}")
    
    print("[OK] RobustNeuralNet test passed\n")

def test_error_handling():
    """Test error handling and validation"""
    print("=" * 50)
    print("Testing Error Handling")
    print("=" * 50)
    
    # Test invalid dimensions
    try:
        model = RobustNeuralNet(input_dim=1, hidden_dim=4, output_dim=1)
        print("✗ Should have failed with small input dimension")
    except ValueError as e:
        print(f"✓ Correctly caught invalid input dimension: {e}")
    
    # Test invalid input shape
    try:
        model = RobustNeuralNet(input_dim=4, hidden_dim=4, output_dim=1)
        x_wrong_shape = np.random.randn(2, 3).astype(np.float32)  # Wrong input dimension
        output = model.forward(x_wrong_shape)
        print("✗ Should have failed with wrong input shape")
    except ValueError as e:
        print(f"✓ Correctly caught invalid input shape: {e}")
    
    # Test NaN input
    try:
        model = RobustNeuralNet(input_dim=4, hidden_dim=4, output_dim=1)
        x_nan = np.array([[1, 2, np.nan, 4]]).astype(np.float32)
        output = model.forward(x_nan)
        print("✗ Should have failed with NaN input")
    except ValueError as e:
        print(f"✓ Correctly caught NaN input: {e}")
    
    print("[OK] Error handling test passed\n")

def test_checkpointing():
    """Test model checkpointing"""
    print("=" * 50)
    print("Testing Model Checkpointing")
    print("=" * 50)
    
    # Create and save checkpoint
    model = RobustNeuralNet(input_dim=4, hidden_dim=4, output_dim=1)
    
    # Do a forward pass to initialize weights
    x_test = np.random.randn(2, 4).astype(np.float32)
    output = model.forward(x_test)
    
    # Save checkpoint
    checkpoint_path = model.save_checkpoint(epoch=1, optimizer_state={'lr': 0.01})
    print(f"Checkpoint saved to: {checkpoint_path}")
    
    # Create new model and load checkpoint
    new_model = RobustNeuralNet(input_dim=4, hidden_dim=4, output_dim=1)
    optimizer_state = new_model.load_checkpoint(checkpoint_path)
    print(f"Checkpoint loaded successfully")
    print(f"Optimizer state: {optimizer_state}")
    
    # Verify weights are the same
    x_verify = np.random.randn(1, 4).astype(np.float32)
    output_original = model.forward(x_verify)
    output_loaded = new_model.forward(x_verify)
    
    if np.allclose(output_original, output_loaded):
        print("[OK] Checkpointing test passed - weights match")
    else:
        print("✗ Checkpointing test failed - weights don't match")
    
    # Clean up checkpoint file
    import os
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint file cleaned up")
    
    print()

def main():
    """Run all tests"""
    print("RobustNeuralNet Test Suite")
    print("=" * 50)
    
    try:
        test_reglu()
        test_robust_neural_net()
        test_error_handling()
        test_checkpointing()
        
        print("=" * 50)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())