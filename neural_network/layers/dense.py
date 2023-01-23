import numpy as np

try:
    from .layer import Layer

except ImportError:
    from layer import Layer


class DenseLayer(Layer):

    def __init__(self,n_neurons:int,n_inputs:int,n_outputs:int) -> None:
        super().__init__(n_neurons,n_inputs,n_outputs)
    
    def random_init(self)->None:
        self.weights:np.ndarray = np.random.rand(self.n_neurons,self.n_inputs)
        self.biases:np.ndarray = np.random.rand(self.n_neurons,1)

    def forward(self,inputs:np.ndarray)->np.ndarray:
        self.inputs:np.ndarray = inputs
        return np.dot(self.weights,inputs) + self.biases
    
    def backward(self,output_grad:np.ndarray,learning_rate:float)->np.ndarray:
        #TODO: Implement backward
        raise NotImplementedError

    def load(self)->None:
        #TODO Implement load weights and biases
        raise NotImplementedError
        
    def save(self,path:str)->None:
        #TODO Implement save weights and biases
        raise NotImplementedError

if __name__ == "__main__":
    d:DenseLayer = DenseLayer(1,1,1)
    d.random_init()
    print(d.weights)
    print(d.biases)
    print(d.forward(np.array([1])))
