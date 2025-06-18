import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def get_data(random=True):
    np.random.seed()
    if random:
        X = np.random.uniform(low=-10, high=10, size=(400, 2))
    else:
        data1 = np.random.randn(100, 2) + np.array([5, 5])
        data2 = np.random.randn(100, 2) + np.array([-5, -5])
        data3 = np.random.randn(100, 2) + np.array([5, -5])
        data4 = np.random.randn(100, 2) + np.array([-5, 5])
        X = np.vstack([data1, data2, data3, data4])
    return X

def main():
    X = get_data()
    # Apply KMeans from sklearn
    kmeans = KMeans(n_clusters=6, n_init=10)
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

if __name__ == "__main__":
    main()
