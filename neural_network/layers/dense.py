import numpy as np

try:
    from .layer import Layer

except ImportError:
    from layer import Layer


class DenseLayer(Layer):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__(n_inputs, n_outputs)

    def random_init(self) -> None:
        self.weights: np.ndarray = np.random.rand(self.n_inputs, self.n_outputs) - 0.5
        self.biases: np.ndarray = np.random.rand(1, self.n_outputs) - 0.5

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs: np.ndarray = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, output_grad: np.ndarray, learning_rate: float) -> np.ndarray:
        weights_grad = np.dot(self.inputs.T, output_grad)
        inp_grad = np.dot(output_grad, self.weights.T)
        # adapt weights and biases
        self.weights -= learning_rate * weights_grad
        self.biases -= learning_rate * output_grad
        return inp_grad

    def load(self) -> None:
        # TODO Implement load weights and biases
        raise NotImplementedError

    def save(self, path: str) -> None:
        # TODO Implement save weights and biases
        raise NotImplementedError
