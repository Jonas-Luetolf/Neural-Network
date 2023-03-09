import numpy as np

try:
    from .activation import ActivationFunction

except ImportError:
    from activation import ActivationFunction


class ReLu(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(lambda x: np.maximum(x, 0), lambda x: np.heaviside(x, 0))


if __name__ == "__main__":
    r = ReLu()
    print(r.forward(np.array([1])))
    r.backward(np.array([0]))
