import numpy as np

from .activation import ActivationFunction


class Tanh(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(lambda x: np.tanh(x), lambda x: 1 - (np.tanh(x) ** 2))
