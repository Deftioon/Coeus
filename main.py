import coeus.coeus as coeus

X = coeus.linalg.tensor([1, 2, 3])
y = coeus.linalg.tensor([1, 2, 3])

model = coeus.regression.Linear([lambda x: x], "MSE")
model.fit(X, y)
print(model)