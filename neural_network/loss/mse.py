import numpy as np
from .loss_function import LossFunction


class MSE(LossFunction):
    def __init__(self) -> None:
        super().__init__(
            lambda pred, target: np.mean(np.power(target - pred, 2)),
            lambda pred, target: 2 * (pred - target) / np.size(target),
        )


"""        
def mse(pred: np.ndarray, value: np.ndarray):
    return np.mean(np.power(value - pred, 2))


def mse_prime(pred, value):
    return 2 * (pred - value) / np.size(value)

"""
