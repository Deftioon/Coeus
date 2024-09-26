import numpy as np

def covariance(X, S, var):
    return 1/(var ** 2) * X.T @ X + np.linalg.inv(S)

def mean(var, covar, X, y):
    return 1/(var ** 2) * covar @ X.T @ y

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1/(1 + np.exp(-1 * x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def tanh(x):
    return np.tanh(x)

def ddx_relu(x):
    shape = x.shape[1]
    arr = np.zeros((shape, shape))
    for i in range(shape):
        arr[i][i] = np.where(x > 0, 1, 0)[0][i]
    return arr

def ddx_sigmoid(x):
    shape = x.shape[1]
    arr = np.zeros((shape, shape))
    for i in range(shape):
        arr[i][i] = sigmoid(x)[0][i] * (1 - sigmoid(x)[0][i])
    return arr

# TODO: Implement Softmax Derivative
# TODO: Fix Jacobian Activation
def ddx_softmax(x):
    pass

def ddx_tanh(x):
    shape = x.shape[1]
    arr = np.zeros((shape, shape))
    for i in range(shape):
        arr[i][i] = 1 - tanh(x)[0][i] ** 2
    return arr

def MSE(x, y):
    return np.mean((x - y) ** 2)

def ddx_MSE(x, y):
    return 2 * (x - y)

def cross_entropy(x, y):
    return -np.sum(y * np.log(x))

def ddx_cross_entropy(x, y):
    return x - y