import numpy as np
from coeus.models.neural import Model
from coeus.models import functions
from coeus.data.preprocessing import *

x = np.sort(np.random.rand(100, 1, 1) / 10, axis = 0)
x = normalize(x)
y = x ** 2

print(x)

model = Model()
model.Input(1)
model.Hidden(2, "relu")
model.Hidden(1, "relu")

model.init()
model.train(x, y, 0.01, "MSE", 50)
model.plot_loss()
