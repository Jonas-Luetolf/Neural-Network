import numpy as np

try:
    from .activation import ActivationFunction

except ImportError:
    from activation import ActivationFunction


class Tanh(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(lambda x: np.tanh(x), lambda x: 1-(np.tanh(x)**2))



if __name__ == "__main__":
    t=Tanh()
    print(t.forward([[1],[-1],[2],[-2]]))
