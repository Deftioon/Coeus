import numpy as np

if __name__ == "__main__":
    from exception import *
    import functions

else:
    from coeus.models.exception import *
    import coeus.models.functions as functions

activations = {
        "linear": lambda x: x,
        "relu": functions.relu,
        "sigmoid": functions.sigmoid,
        "tanh": functions.tanh
    }

activations_der = {
        "linear": 1,
        "relu": functions.ddx_relu,
        "sigmoid": functions.ddx_sigmoid,
        "tanh": functions.ddx_tanh
    }

def get_act(activation):
    return activations[activation]

def get_der(activation):
    return activations_der[activation]

class Model:
    def __init__(self):
        self.layers = []
        self.layer_num = 0
    
    def __str__(self):
        self.init()
        return f"""
=============MODEL SUMMARY==============
Neural Network:
Input Size: {self.layers[0].layer_size}
Output Size: {self.layers[-1].layer_size}
===============DETAILS==================
Parameters: {sum([i.weights.shape[0] * i.weights.shape[1] for i in self.layers[1:]])}
Layer Sizes: {[layer.layer_size for layer in self.layers]}
Activations: {[layer.activation for layer in self.layers]}
Weight Sizes: {[layer.weights.shape for layer in self.layers[1:]]}
=======================================
"""
    
    def Input(self, dims):
        self.layers.append(Input(dims))
        self.layer_num += 1

    def Hidden(self, layer_size, activation):
        self.layers.append(Hidden(layer_size, activation))
        self.layer_num += 1
    
    def init(self):
        for i in range(1, self.layer_num):
            self.layers[i].weights = np.random.rand(self.layers[i - 1].layer_size, self.layers[i].layer_size)
    
    def forward(self, data):
        if data.shape[1] != self.layers[0].layer_size:
            raise ShapeError(data.shape[1], self.layers[0].layer_size)
        
        self.layers[0].z = data
        self.layers[0].a = data
        for i in range(1, self.layer_num):
            self.layers[i].forward(self.layers[i - 1])
        
        return self.layers[-1].a


class Input:
    def __init__(self, dims):
        self.layer_size = dims
        self.activation = "linear"
        self.weights = None
        self.a = None
        self.z = None
        
class Hidden:
    def __init__(self, layer_size, activation):
        self.layer_size = layer_size
        self.activation = activation
        self.weights = None
        self.a = None
        self.z = None

    def forward(self, layer):
        self.z = layer.a @ self.weights
        self.a = get_act(self.activation)(self.z)

    def backward(self, layer):
        pass