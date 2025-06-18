import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample data: 3 clusters
np.random.seed()

cluster1 = np.random.randn(100, 2) + np.array([5, 5])
cluster2 = np.random.randn(100, 2) + np.array([-5, -5])
cluster3 = np.random.randn(100, 2) + np.array([5, -5])
data4 = np.random.randn(100, 2) + np.array([-5, 5])
X = np.vstack([cluster1, cluster2, cluster3, data4])

# Apply KMeans from sklearn
kmeans = KMeans(n_clusters=4, n_init=10)
kmeans.fit(X)

# Get results
labels = kmeans.labels_  # which cluster each point belongs to
centroids = kmeans.cluster_centers_  # final centroid positions

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.6)
plt.scatter(
    centroids[:, 0], centroids[:, 1], color="red", marker="X", s=200, label="Centroids"
)
plt.title("K-Means Clustering (sklearn)")
plt.legend()
plt.show()
