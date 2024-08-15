import numpy as np
import h5py

try:
    from .layer import Layer

except ImportError:
    from layer import Layer


class DenseLayer(Layer):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__(n_inputs, n_outputs)

    def random_init(self) -> None:
        self.weights: np.ndarray = np.random.rand(self.n_outputs, self.n_inputs) - 0.5
        self.biases: np.ndarray = np.random.rand(self.n_outputs, 1) - 0.5

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs: np.ndarray = inputs
        return np.dot(self.weights, inputs) + self.biases

    def backward(self, output_grad: np.ndarray, learning_rate: float) -> np.ndarray:
        weights_grad = np.dot(output_grad, self.inputs.T)
        inp_grad = np.dot(self.weights.T, output_grad)

        # adapt weights and biases
        self.weights -= learning_rate * weights_grad
        self.biases -= learning_rate * output_grad
        return inp_grad

    def save(self, group_name: str, h5file: h5py.File) -> None:
        group = h5file.create_group(group_name)
        group.create_dataset("weights", data=self.weights)
        group.create_dataset("biases", data=self.biases)

    def load(self, group: h5py.Group) -> None:
        self.weights = np.array(group["weights"])
        self.biases = np.array(group["biases"])
