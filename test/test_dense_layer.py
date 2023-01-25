import unittest
import numpy as np

from neural_network.layers import DenseLayer

class TestDenseLayer(unittest.TestCase):

    def setUp(self):
        self.d = DenseLayer(1,1)
        self.d.random_init()
    
    def test_dense_init(self):
        self.assertEqual(np.shape(self.d.weights),(1,1),"wrong weights size")
        self.assertEqual(np.shape(self.d.biases),(1,1),"wrong biases size")

    def test_dense_forward(self):
        self.assertEqual((self.d.weights+self.d.biases), self.d.forward(np.array([1])),"incorect forward method")

    def test_dense_backward(self):
        # TODO implement testing for DenseLayer backward
        pass
