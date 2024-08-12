# a basic Layer class used by all Layer types
from abc import abstractmethod
import numpy as np
import h5py


class Layer:
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        self.n_inputs: int = n_inputs
        self.n_outputs: int = n_outputs

    def __repr__(self) -> str:
        return f"{type(self)}({self.n_inputs}, {self.n_outputs})"

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output_grad: np.ndarray, learning_rate: float) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, group_name: str, h5file: h5py.File) -> None:
        pass

    @abstractmethod
    def load(self, group: h5py.Group) -> None:
        pass
