from collections.abc import Callable
import numpy as np


class ActivationFunction:
    def __init__(
        self, function: Callable, function_derivative: Callable
    ) -> None:
        self.function: Callable = function
        self.function_derivative: Callable = function_derivative

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs: np.ndarray = inputs
        return self.function(inputs)

    def backward(
        self, output_grad: np.ndarray, learning_rate: float = None
    ) -> np.ndarray:
        return np.multiply(output_grad, self.function_derivative(self.inputs))
