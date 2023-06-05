class LossFunction:
    def __init__(self, function, function_prime):
        self.function = function
        self.function_prime = function_prime

    def loss(self, pred, target):
        return self.function(pred, target)

    def loss_prime(self, pred, target):
        return self.function_prime(pred, target)
