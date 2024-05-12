import numpy as np
import coeus.models.regression as regression
import coeus.models.kmeans as kmeans

# KMeans
X = np.random.randn(1000, 2)
kmeans_model = kmeans.KMeans(6, 100)
kmeans_model.fit(X)
print(kmeans_model)
kmeans_model.show(save = True, custom_title = "KMeans 100")