import numpy as np

from neural_network.layers import DenseLayer
from neural_network.activation import Tanh


def main():
    network = [DenseLayer(2, 2), Tanh(), DenseLayer(2, 1), Tanh()]
    for layer in network:
        try:
            layer.random_init()

        except:
            pass

    inputs = [
        np.array([[0], [0]]),
        np.array([[1], [0]]),
        np.array([[0], [1]]),
        np.array([[1], [1]]),
    ]
    outputs = [0, 1, 1, 0]

    for inp, out in zip(inputs, outputs):
        x = inp
        for layer in network:
            x = layer.forward(x)

        print(x)


if __name__ == "__main__":
    main()
