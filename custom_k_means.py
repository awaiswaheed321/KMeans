import numpy as np
import matplotlib.pyplot as plt


def initialize_centroids(X, k):
    """Randomly pick k data points as initial centroids"""
    indices = np.random.choice(a=len(X), size=k, replace=False)
    print(indices)
    return X[indices]


def assign_clusters(X, centroids):
    """Assign each point to the closest centroid"""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, k):
    """Recompute the centroids as the mean of points in each cluster"""
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids


def kmeans(X, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, k)

    for i in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)

        # Check convergence
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids

    return centroids, labels

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
    X = get_data(True)

    # Plot initial data
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.title("Unlabeled Data")
    # plt.show()

    k = 4
    centroids, labels = kmeans(X, k)

    # Plot final clusters
    for i in range(k):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f"Cluster {i}")
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        color="black",
        marker="x",
        s=200,
        label="Centroids",
    )
    plt.legend()
    plt.title("K-Means Clustering Result")
    plt.show()


if __name__ == "__main__":
    main()
