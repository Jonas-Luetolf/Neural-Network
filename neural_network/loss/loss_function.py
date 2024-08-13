from typing import Callable

import numpy as np


class LossFunction:
    def __init__(self, function: Callable, function_prime: Callable):
        self.function: Callable = function
        self.function_prime: Callable = function_prime

    def loss(self, pred: np.ndarray, target: np.ndarray):
        return self.function(pred, target)

    def loss_prime(self, pred: np.ndarray, target: np.ndarray):
        return self.function_prime(pred, target)
