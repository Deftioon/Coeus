import numpy as np
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k: int):
        self.k = k
        self.X = None
        self.y = None

    def __str__(self):
        return f""""""
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def predict(self, X: np.ndarray):
        distances = np.linalg.norm(self.X[:, None] - X, axis=2)
        nearest = np.argsort(distances, axis=0)[:self.k]
        return np.mean(self.y[nearest], axis=0)
    
    def show(self, custom_title = "KNN", custom_xlabel = "X", custom_ylabel = "Y", save = False):
        plt.title(custom_title)
        plt.xlabel(custom_xlabel)
        plt.ylabel(custom_ylabel)

        plt.scatter(self.X, self.y)
        plt.show()