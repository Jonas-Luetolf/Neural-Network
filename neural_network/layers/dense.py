import numpy as np

try:
    from .layer import Layer

except ImportError:
    from layer import Layer


class DenseLayer(Layer):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__(n_inputs, n_outputs)

    def random_init(self) -> None:
        self.weights: np.ndarray = np.random.rand(self.n_outputs, self.n_inputs)
        self.biases: np.ndarray = np.random.rand(self.n_outputs, 1)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs: np.ndarray = inputs
        return np.dot(self.weights, inputs) + self.biases

    def backward(self, output_grad: np.ndarray, learning_rate: float) -> np.ndarray:
        weights_grad = output_grad * self.inputs.T

        # adapt weights and biases
        self.weights = learning_rate * weights_grad
        self.biases = learning_rate * output_grad

        return weights_grad.T * output_grad

    def load(self) -> None:
        # TODO Implement load weights and biases
        raise NotImplementedError

    def save(self, path: str) -> None:
        # TODO Implement save weights and biases
        raise NotImplementedError


if __name__ == "__main__":
    d: DenseLayer = DenseLayer(2, 2)
    d.random_init()
    print(d.weights)
    print(d.biases)

    print("")
