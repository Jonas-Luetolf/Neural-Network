import numpy as np

from .activation import ActivationFunction


class ReLu(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(lambda x: np.maximum(x, 0), lambda x: np.heaviside(x, 0))
