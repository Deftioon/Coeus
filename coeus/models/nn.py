import numpy as np
from coeus.models.exception import *
import coeus.models.functions as functions

class FCN:
    def __init__(self):
        self.supported_activation = ["linear", "relu", "sigmoid", "tanh"]
        self.supported_optimizer = ["SGD"]
        self.supported_loss = ["MSE", "CrossEntropy"]

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.gradients_z = None
        self.gradients_w = None

        self.layers = []

        self.layer_num = 0
        self.layer_dims = []
        self.hidden_num = 0

        self.weights = []
        self.activations = []

        self.a = []
        self.z = []

        self.output = None

        self.average_loss = "Not Calculated"
        self.average_test_confidence = "Not Calculated"

    def __str__(self):
        return f"""
=============MODEL SUMMARY==============
Fully Connected Neural Network:
Input Size: {self.layer_dims[0]}
Output Size: {self.layer_dims[-1]}
===============DETAILS==================
Parameters: {sum([i.shape[0] * i.shape[1] for i in self.weights])}
Layer Sizes: {self.layer_dims}
Activations: {self.activations}
Weight Sizes: {[i.shape for i in self.weights]}

Average Train Loss: {self.average_loss}
Average Test Accuracy: {self.average_test_confidence}
=======================================
"""
    
    def activate(self, x, activation: str):
        if activation == "linear":
            return functions.linear(x)
        elif activation == "relu":
            return functions.relu(x)
        elif activation == "sigmoid":
            return functions.sigmoid(x)
        elif activation == "tanh":
            return functions.tanh(x)
        elif activation == "softmax":
            return functions.softmax(x)
    
    def loss(self, x, y, loss: str):
        if loss == "MSE":
            return functions.MSE(x, y)
        elif loss == "CrossEntropy":
            return functions.cross_entropy(x, y)
    
    def ddx_activation(self, x, activation: str):
        if activation == "linear":
            return functions.linear(x)
        elif activation == "relu":
            return functions.ddx_relu(x)
        elif activation == "sigmoid":
            return functions.ddx_sigmoid(x)
        elif activation == "tanh":
            return functions.ddx_tanh(x)
        elif activation == "softmax":
            return functions.ddx_softmax(x)
    
    def ddx_loss(self, x, y, loss: str):
        if loss == "MSE":
            return functions.ddx_MSE(x, y)
        elif loss == "CrossEntropy":
            return functions.ddx_cross_entropy(x, y)

    def Input(self, dims: int):
        self.layer_dims.append(dims)
        self.layer_num += 1

    def Hidden(self, dims: int, activation: str):
        if activation not in self.supported_activation:
            raise ActivationError(activation, self.supported_activation)

        self.layer_dims.append(dims)
        self.layer_num += 1
        self.hidden_num += 1

        self.weights.append(np.random.rand(self.layer_dims[-2], self.layer_dims[-1]))
        self.activations.append(activation)

    def Output(self, dims: int, activation: str):
        if activation not in self.supported_activation:
            raise ActivationError(activation, self.supported_activation)

        self.layer_dims.append(dims)
        self.layer_num += 1
        self.hidden_num += 1

        self.weights.append(np.random.rand(self.layer_dims[-2], self.layer_dims[-1]))
        self.activations.append(activation)

    def load(self, X_train, y_train, X_test, y_test):
        if X_train.shape[1] != self.layer_dims[0]:
            raise ShapeError(X_train.shape[1], self.layer_dims[0])

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train(self, epochs, learning_rate, loss, optimizer):
        if loss not in self.supported_loss:
            raise NotImplementedError(f"Loss function {loss} not supported. Supported functions are {self.supported_loss}")

        for epoch in range(epochs):
            self.gradients_w = []
            self.gradients_z = []
            self.layers = []
            for i, j in zip(self.X_train, self.y_train):
                z_i = i.copy()
                for k in range(self.layer_num - 1):
                    z_i = z_i @ self.weights[k]
                    a_i = self.activate(z_i, self.activations[k])

                    self.layers.append(a_i)