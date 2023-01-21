import numpy as np

try:
    from .activation import ActivationFunction

except ImportError:
    from activation import ActivationFunction


class ReLu(ActivationFunction):
    def __init__(self,function) -> None:
        super().__init__(function)
