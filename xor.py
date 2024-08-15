import numpy as np
import sys

from neural_network.layers import DenseLayer
from neural_network.activation import Tanh
from neural_network.loss import MSE
from neural_network.network import NeuralNetwork

TRAIN = False
PATH = "./xor.hdf5"


def main():
    network = NeuralNetwork(MSE())
    network.add_layer(DenseLayer(2, 3))
    network.add_layer(Tanh())
    network.add_layer(DenseLayer(3, 3))
    network.add_layer(Tanh())
    network.add_layer(DenseLayer(3, 1))
    network.add_layer(Tanh())

    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2, 1)
    outputs = np.array([[0], [1], [1], [0]]).reshape(4, 1, 1)

    if TRAIN:
        print(f"neural network gets trained and saved to {PATH}")

        network.random_init()
        network.train(inputs, outputs, 1000, 0.1)
        network.save(PATH)

    else:
        print(f"neural network loaded from {PATH}")
        network.load(PATH)

    for inp, out in zip(inputs, outputs):
        pred = np.around(network.forward(inp), 0)
        print(pred[0][0], " ", pred[0][0] == out[0][0])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        TRAIN = True
    main()
