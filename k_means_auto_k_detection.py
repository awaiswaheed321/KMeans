import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator


def get_data(random=True):
    np.random.seed()
    if random:
        X = np.random.uniform(low=-10, high=10, size=(1600, 2))
    else:
        data1 = np.random.randn(400, 2) + np.array([5, 5])
        data2 = np.random.randn(400, 2) + np.array([-5, -5])
        data3 = np.random.randn(400, 2) + np.array([5, -5])
        data4 = np.random.randn(400, 2) + np.array([-5, 5])
        X = np.vstack([data1, data2, data3, data4])
    return X


def find_optimal_k(X, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Automatically find the "elbow"
    kl = KneeLocator(range(1, max_k + 1), wcss, curve="convex", direction="decreasing")
    optimal_k = kl.elbow

    # Plot the elbow graph
    plt.plot(range(1, max_k + 1), wcss, marker="o")
    plt.axvline(
        x=optimal_k, color="red", linestyle="--", label=f"Elbow at k={optimal_k}"
    )
    plt.title("Elbow Method to Find Optimal k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS")
    plt.legend()
    plt.show()

    return optimal_k


def main():
    X = get_data(True)

    # Step 1: Determine optimal number of clusters
    optimal_k = find_optimal_k(X)

    # Step 2: Apply KMeans using optimal k
    kmeans = KMeans(n_clusters=optimal_k, n_init=10)
    kmeans.fit(X)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Step 3: Visualize clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.6)
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        color="red",
        marker="X",
        s=200,
        label="Centroids",
    )
    plt.title(f"K-Means Clustering (k={optimal_k})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
