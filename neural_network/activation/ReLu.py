import numpy as np

try:
    from .activation import ActivationFunction

except ImportError:
    from activation import ActivationFunction


class ReLu(ActivationFunction):
    def __init__(self) -> None:
        #TODO implement ReLu function
        #super().__init__(<ReLu function>)
        raise NotImplementedError
