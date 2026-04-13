import numpy as np

class ReGLU:
    """reGLU activation function: x * ReLU(x) where x is split into two halves"""

    def __init__(self):
        self.cache_input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache_input = x
        split_idx = x.shape[-1] // 2
        x1 = x[..., :split_idx]
        x2 = x[..., split_idx:]
        output = x1 * np.maximum(0, x2)
        return output.astype(np.float32)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.cache_input is None:
            raise ValueError("Forward pass must be called before backward pass")
        x = self.cache_input
        split_idx = x.shape[-1] // 2
        x1 = x[..., :split_idx]
        x2 = x[..., split_idx:]
        grad_x1 = grad_output * (x2 > 0).astype(np.float32)
        grad_x2 = grad_output * x1 * (x2 > 0).astype(np.float32)
        grad_input = np.concatenate([grad_x1, grad_x2], axis=-1)
        return grad_input

class Dense:
    """Dense layer"""
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        std = np.sqrt(2.0 / input_dim)
        self.weights = np.random.randn(input_dim, output_dim).astype(np.float32) * std
        self.bias = np.zeros(output_dim, dtype=np.float32)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self.cache_input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache_input = x
        return np.dot(x, self.weights) + self.bias

class RobustNeuralNet:
    """Robust Neural Network"""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 10, output_dim: int = 1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = [
            Dense(input_dim, hidden_dim),
            ReGLU(),
            Dense(hidden_dim, output_dim),
        ]
        self.gradient_cache = {}
        self.input_cache = None
        self.output_cache = None
