import numpy as np

from .loss_function import LossFunction


class MSE(LossFunction):
    def __init__(self) -> None:
        super().__init__(
            lambda pred, target: np.mean(np.power(target - pred, 2)),
            lambda pred, target: 2 * (pred - target) / np.size(target),
        )
