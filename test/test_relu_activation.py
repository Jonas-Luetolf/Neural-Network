import unittest
import numpy as np

from neural_network.activation import ReLu


class TestReLu(unittest.TestCase):
    def setUp(self):
        self.activation_function = ReLu()

    def test_ReLu_forward(self):
        self.assertEqual(self.activation_function.forward(np.array([1])), np.array([1]))
        self.assertEqual(
            self.activation_function.forward(np.array([-1])), np.array([0])
        )
