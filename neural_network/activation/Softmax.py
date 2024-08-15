import numpy as np

from .activation import ActivationFunction


class Softmax(ActivationFunction):
    def __init__(self) -> None:
        pass

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.exp(inputs) / np.sum(np.exp(inputs))
        return self.output

    def backward(
        self, output_grad: np.ndarray, learning_rate: float = None
    ) -> np.ndarray:
        n = np.size(self.output)

        return np.dot((np.identity(n) - self.output.T) * self.output, output_grad)
