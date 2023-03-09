# a basic Layer class used by all Layer types
class Layer:
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        self.n_inputs: int = n_inputs
        self.n_outputs: int = n_outputs

    def __repr__(self) -> str:
        return f"{type(self)}({self.n_inputs}, {self.n_outputs})"
