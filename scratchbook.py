import numpy as np


def clustering():
    # 3 data points
    X = np.array([[1, 2], [4, 5], [8, 9]])  # point A  # point B  # point C

    # 2 centroids
    centroids = np.array([[1, 1], [7, 8]])  # cluster 0  # cluster 1

    print("\n\n*******************************")
    print("X shape:", X.shape)
    print("X =", X)

    X_expanded = X[:, np.newaxis]
    print("\n\n*******************************")
    print("\n\nX[:, np.newaxis] shape:", X_expanded.shape)
    print("X[:, np.newaxis] =", X_expanded)

    diff = X_expanded - centroids
    print("\n\n*******************************")
    print("Shape:", diff.shape)
    print("Difference array:\n", diff)

    distances = np.linalg.norm(diff, axis=2)
    print("\n\n*******************************")
    print("Distances:\n", distances)

    print("\n\n*******************************")
    labels = np.argmin(distances, axis=1)
    print("Cluster assignments:", labels)
