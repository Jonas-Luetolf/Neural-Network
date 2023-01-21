import numpy as np

try:
    from .layer import Layer

except ImportError:
    from layer import Layer


class DenseLayer(Layer):
    def __init__(self,n_neurons:int,n_inputs:int,n_outputs:int) -> None:
        super().__init__(n_neurons,n_inputs,n_outputs)
    
    def random_init(self)->None:
        self.weights = np.random.rand(self.n_neurons,self.n_inputs)
        self.biases = np.random.rand(self.n_neurons,1)

    def forward(self,inputs:np.ndarray)->np.ndarray:
        self.inputs = inputs
        return np.dot(self.weights,inputs) + self.biases
    
    def backward(self,output_grad:float,learning_rate:float):
        #TODO: Implement backward
        pass

    def load(self)->None:
        pass

if __name__ == "__main__":
    d = DenseLayer(1,1,1)
    d.random_init()
    print(d.weights)
    print(d.biases)
    print(d.forward(np.array([1])))
