import numpy as np
import coeus.models.regression as regression

model = regression.Linear([lambda x: x, lambda x: x**2], "MSE")

x = np.sort(np.random.rand(100))
y = 0.5 * x ** 2 + 0.1 * x + np.random.rand(100)

model.fit(x, y)

print(model.forecast(5,0.1))