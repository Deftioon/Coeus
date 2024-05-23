import numpy as np
from coeus.models.neural import Model
from coeus.models import functions
from coeus.data.preprocessing import *

x = np.sort(np.random.rand(100, 1, 10), axis=0)
x = normalize(x)
y = x**2

print(x.shape)
print(y.shape)

model = Model()
model.Input(10)
model.Hidden(8, "tanh")
model.Hidden(4, "relu")
model.Hidden(10, "tanh")

model.init()
model.train(x, y, x, y, 0.01, "MSE", 0.5, 100)
model.plot_loss()