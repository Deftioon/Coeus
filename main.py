import numpy as np
import coeus.models.regression as regression

# Generate Random Data
X = np.sort(np.random.rand(100) * 10)
Y = np.cos(X) + np.random.rand(100)

# Create Regression Model
model = regression.Regression([lambda x: np.cos(x)], "MSE")
model.fit(X, Y)

print(model)

model.show(30, 0.1, "Regression", "X", "Y", True)