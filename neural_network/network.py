import numpy as np
import h5py

from .loss.loss_function import LossFunction
from .layers.layer import Layer
from .activation.activation import ActivationFunction


class NeuralNetwork:
    def __init__(self, loss_function: LossFunction) -> None:
        self.layers: list[Layer | ActivationFunction] = []
        self.loss_function = loss_function

    def add_layer(self, layer: Layer | ActivationFunction) -> None:
        self.layers.append(layer)

    def random_init(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "random_init"):
                layer.random_init()

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss: np.ndarray, lr: float) -> None:
        x = loss
        for layer in self.layers[::-1]:
            x = layer.backward(x, lr)

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float) -> None:
        for _ in range(epochs):
            for x, y in zip(X, Y):
                x = self.forward(x)
                loss = self.loss_function.loss_prime(x, y)
                self.backward(loss, lr)

    def save(self, path: str) -> None:
        with h5py.File(path, "w") as h5file:
            for i, layer in enumerate(self.layers):
                if isinstance(layer, Layer):
                    layer.save(f"layer_{i}", h5file)

    def load(self, path: str) -> None:
        with h5py.File(path, "r") as h5file:
            for i, layer in enumerate(self.layers):
                if isinstance(layer, Layer):
                    group_name = f"layer_{i}"
                    if group_name not in h5file:
                        raise KeyError(f"Group {group_name} not found in the file.")

                    group = h5file[group_name]

                    if not isinstance(group, h5py.Group):
                        raise ValueError(
                            f"Expected Group, but found {type(group).__name__}"
                        )
                    layer.load(group)
