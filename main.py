import numpy as np
from coeus.models.neural import Model
from coeus.models import functions
from coeus.data.preprocessing import *

x = np.array([
    np.array([[0,1]]),
    np.array([[1,0]]),
    np.array([[0,0]]),
    np.array([[1,1]])
])

y = np.array([
    np.array([[0]]),
    np.array([[1]]),
    np.array([[1]]),
    np.array([[0]])
])

print(x.shape)
print(y.shape)

model = Model()
model.Input(2)
model.Hidden(2, "relu")
model.Hidden(2, "sigmoid")
model.Hidden(1, "relu")

model.init()
model.train(x, y, x, y, 0.01, "cross_entropy", 0.5, 1000)
model.plot_loss()