import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k: int, max_iter: int):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        self.X = None

    def __str__(self):
        return f"""
=============MODEL SUMMARY==============
K-Means Model:
Iterations: {self.max_iter}
Centroids: {self.k}
===============DETAILS==================
Average Distance: {np.mean(np.linalg.norm(self.X[:, None] - self.centroids, axis=2))}
=======================================
"""
    
    def fit(self, X: np.ndarray):
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        self.X = X

        for i in range(self.max_iter):
            self.labels = np.argmin(np.linalg.norm(X[:, None] - self.centroids, axis=2), axis=1)
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])

            if np.all(new_centroids == self.centroids):
                break

            self.centroids = new_centroids
    
    def predict(self, X: np.ndarray):
        return np.argmin(np.linalg.norm(X[:, None] - self.centroids, axis=2), axis=1)
    
    def show(self, custom_title = "KMeans", custom_xlabel = "X", custom_ylabel = "Y", save = False):
        plt.title(custom_title)
        plt.xlabel(custom_xlabel)
        plt.ylabel(custom_ylabel)

        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.labels)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', color='r')

        if save:
            plt.savefig(f"exports/{custom_title}.png")

        plt.show()