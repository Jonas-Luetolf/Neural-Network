from types import FunctionType
import numpy as np

class ActivationFunction:
    def __init__(self,function:FunctionType) -> None:
        self.function:FunctionType = function

    def forward(self,inputs:np.ndarray)->np.ndarray:
        self.inputs:np.ndarray = inputs
        return self.function(inputs)

    def backward(self,output_grad:np.ndarray)->np.ndarray:
        #TODO implement backward for ActivationFunction
        raise NotImplementedError
