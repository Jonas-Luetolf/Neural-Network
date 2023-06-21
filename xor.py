import numpy as np

from neural_network.layers import DenseLayer
from neural_network.activation import Tanh, ReLu
from neural_network.loss import MSE


def main():
    mse = MSE()
    network = [
        DenseLayer(2, 3),
        Tanh(),
        DenseLayer(3, 3),
        Tanh(),
        DenseLayer(3, 1),
        Tanh(),
    ]
    for layer in network:
        try:
            layer.random_init()

        except:
            pass

    inputs = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    outputs = np.array([[[0]], [[1]], [[1]], [[0]]])
    print(outputs.shape)
    print(inputs.shape)
    for i in range(0, 1000):
        for inp, out in zip(inputs, outputs):
            x = inp
            for layer in network:
                x = layer.forward(x)
            loss = mse.loss_prime(x, out)
            print(loss)
            for layer in reversed(network):
                loss = layer.backward(loss, 0.1)

    x = np.array([[0, 1]])
    for layer in network:
        x = layer.forward(x)
    print(np.around(x, 3))

    x = np.array([[1, 1]])
    for layer in network:
        x = layer.forward(x)
    print(np.around(x, 3))

    x = np.array([[1, 0]])

    for layer in network:
        x = layer.forward(x)
    print(np.around(x, 3))


if __name__ == "__main__":
    main()
