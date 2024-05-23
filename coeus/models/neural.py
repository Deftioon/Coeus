import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

loss = {
    "MSE": functions.MSE,
    "cross_entropy": functions.cross_entropy
}

loss_der = {
    "MSE": functions.ddx_MSE,
    "cross_entropy": functions.ddx_cross_entropy
}

def get_act(activation):
    return activations[activation]

def get_act_der(activation):
    return activations_der[activation]

def get_loss(loss_func):
    return loss[loss_func]

def get_loss_der(loss_func):
    return loss_der[loss_func]

class Model:
    def __init__(self):
        self.layers = []
        self.layer_num = 0

        self.loss_list = []
        self.val_loss_list = []
    
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
        self.layers[-1].layer_num = self.layer_num
        self.layer_num += 1

    def Hidden(self, layer_size, activation):
        self.layers.append(Hidden(layer_size, activation))
        self.layers[-1].layer_num = self.layer_num
        self.layer_num += 1
    
    def init(self):
        for i in range(1, self.layer_num):
            self.layers[i].weights = np.random.rand(self.layers[i - 1].layer_size, self.layers[i].layer_size)/10
    
    def forward(self, data):
        if data.shape[1] != self.layers[0].layer_size:
            raise ShapeError(data.shape[1], self.layers[0].layer_size)
        
        self.layers[0].z = data
        self.layers[0].a = data
        for i in range(1, self.layer_num):
            self.layers[i].forward(self.layers[i - 1])
        
        return self.layers[-1].a

    def backward(self, data, target, lr, loss):
        if data.shape[1] != self.layers[0].layer_size:
            raise ShapeError(data.shape[1], self.layers[0].layer_size)
        
        for i in range(self.layer_num - 1, 0, -1):
            if i == self.layer_num - 1:
                self.layers[i].backward(lr, self.layers[i], self.layers[i-1], True, loss, target)
            else:
                self.layers[i].backward(lr, self.layers[i + 1], self.layers[i - 1], False, loss, target)

    def update(self):
        for i in range(1, self.layer_num):
            self.layers[i].update()
    
    def calc_loss(self, target, loss):
        return get_loss(loss)(self.layers[-1].a, target)

    def train(self, data, target, val_x, val_y, lr, loss, aim, epochs):
        print(data.shape[0])
        for epoch in range(epochs):
            epoch_data = {
                "loss": [],
                "val_loss": [],
                "confusion": np.zeros((data.shape[2] + 1, data.shape[2] + 1))
            }
            for x_val, y_val in zip(val_x, val_y):
                self.forward(x_val)
                output = np.argmax(self.layers[-1].a)
                targ = np.argmax(y_val)

                epoch_data["val_loss"].append(self.calc_loss(y_val, loss))
            
            for x_train, y_train in tqdm(zip(data, target)):
                self.forward(x_train)
                self.backward(x_train, y_train, lr, loss)
                self.update()

                output = np.argmax(self.layers[-1].a)
                targ = np.argmax(y_train)

                epoch_data["loss"].append(self.calc_loss(y_train, loss))
                epoch_data["confusion"][np.argmax(y_train)][targ] += 1 if self.layers[-1].a[0][output] > aim else 0   

            mean_epoch_loss = np.mean(epoch_data["loss"])
            mean_val_loss = np.mean(epoch_data["val_loss"])
            self.loss_list.append(mean_epoch_loss)
            self.val_loss_list.append(mean_val_loss)

            print(f"Epoch [{epoch}/{epochs}] Loss: {mean_epoch_loss: .2f} Validation Loss: {mean_val_loss: .2f}")
            
    
    def plot_loss(self):
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_list, label = "Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.val_loss_list, label = "Validation Loss")
        plt.xlabel("Epoch")
        plt.legend()

        plt.show()



class Input:
    def __init__(self, dims):
        self.layer_size = dims
        self.activation = "linear"
        self.weights = None
        self.layer_num = 0
        self.a = None
        self.z = None
        self.dz = None
        self.dw = None
    
    def __str__(self):
        return str(self.a)
        
class Hidden:
    def __init__(self, layer_size, activation):
        self.layer_size = layer_size
        self.activation = activation
        self.weights = None
        self.new_weights = None
        self.layer_num = 0
        self.a = None
        self.z = None
        self.dz = None
        self.dw = None
    
    def __str__(self):
        return str(self.a)

    def update(self):
        self.weights = self.new_weights if self.new_weights is not None else self.weights

    def forward(self, layer):
        self.z = layer.a @ self.weights
        self.a = get_act(self.activation)(self.z)

    def backward(self, lr, next_layer, prev_layer, last, loss, target):
        self.new_weights = self.weights
        if last:
            self.dz = get_loss_der(loss)(self.a, target) @ get_act_der(self.activation)(self.a)
            self.dw = prev_layer.a.T @ self.dz
            self.new_weights -= lr * self.dw
        
        else:
            self.dz = next_layer.dz @ next_layer.weights.T @ get_act_der(self.activation)(self.a)
            self.dw = prev_layer.a.T @ self.dz
            self.new_weights -= lr * self.dw

class Dropout:
    def __init__(self, drop_rate):
        self.drop_rate = drop_rate

class Conv2D:
    def __init__(self, kernel_size, step, activation):
        self.kernel_size = kernel_size
        self.step = step
        self.activation = activation

class Pooling:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

class Flatten:
    def __init__(self):
        pass

class Recurrent:
    def __init__(self, layer_size, activation):
        self.layer_size = layer_size
        self.activation = activation

class Attention:
    def __init__(self):
        pass