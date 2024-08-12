import numpy as np
import sys

from neural_network.layers import DenseLayer
from neural_network.activation import Tanh
from neural_network.loss import MSE
from neural_network.network import NeuralNetwork

RAND = False
PATH = "./xor.hdf5"


def main():

    network = NeuralNetwork(MSE())
    network.add_layer(DenseLayer(2, 3))
    network.add_layer(Tanh())
    network.add_layer(DenseLayer(3, 3))
    network.add_layer(Tanh())
    network.add_layer(DenseLayer(3, 1))
    network.add_layer(Tanh())

    if RAND:
        print(f"neural network gets trained and saved to {PATH}")
        network.random_init()

        inputs = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
        outputs = np.array([[[0]], [[1]], [[1]], [[0]]])

        network.train(inputs, outputs, 1000, 0.1)
        network.save(PATH)

    else:
        print(f"neural network loaded from {PATH}")
        network.load(PATH)

    x = np.array([[0, 1]])
    y = network.forward(x)
    print("1: ", np.around(y, 0))

    x = np.array([[1, 1]])
    y = network.forward(x)
    print("0: ", np.around(y, 0))

    x = np.array([[1, 0]])
    y = network.forward(x)
    print("1: ", np.around(y, 0))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        RAND = True
    main()
