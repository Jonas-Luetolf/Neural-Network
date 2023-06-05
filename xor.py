import numpy as np

from neural_network.layers import DenseLayer
from neural_network.activation import Tanh, ReLu
from neural_network.loss import MSE


def main():
    mse = MSE()
    network = [DenseLayer(2, 3), Tanh(), DenseLayer(3, 1), ReLu()]
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
    for i in range(0, 10000):
        for inp, out in zip(inputs, outputs):
            x = inp
            for layer in network:
                x = layer.forward(x)

            loss = mse.loss_prime(x, out)
            for layer in reversed(network):
                loss = layer.backward(loss, 0.1)

            # print(x)

    x = np.array([[0], [0]])
    for layer in network:
        x = layer.forward(x)

    print(np.around(x, 3))


if __name__ == "__main__":
    main()
