import numpy as np

from .loss.loss_function import LossFunction
from .layers.layer import Layer
from .activation.activation import ActivationFunction


class NeuralNetwork:
    def __init__(self, loss_function: LossFunction) -> None:
        self.layers = []
        self.loss_function = loss_function

    def add_layer(self, layer: Layer | ActivationFunction) -> None:
        self.layers.append(layer)

    def random_init(self) -> None:
        for layer in self.layers:
            try:
                layer.random_init()
            except AttributeError:
                pass

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
