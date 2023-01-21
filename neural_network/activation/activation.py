from types import FunctionType
import numpy as np

class ActivationFunction:
    def __init__(self,function:FunctionType) -> None:
        self.function = function

    def forward(self,inputs:np.ndarray)->np.ndarray:
        self.inputs = inputs
        return self.function(inputs)
