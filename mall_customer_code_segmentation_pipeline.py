import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator


# Step 1: Load the data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df


# Step 2: Handle missing values (simulate and fill)
def handle_missing(df):
    df_missing = df.copy()
    df_missing.loc[3, "Age"] = np.nan
    df_missing.loc[10, "Gender"] = np.nan
    df_missing["Age"] = df_missing["Age"].fillna(df_missing["Age"].mean())
    df_missing["Gender"] = df_missing["Gender"].fillna(df_missing["Gender"].mode()[0])
    return df_missing


# Step 3: Encode categorical features
def encode_features(df):
    df_encoded = pd.get_dummies(df, columns=["Gender"], drop_first=True)
    return df_encoded


# Step 4: Feature selection
def select_features(df):
    return df.drop(columns=["CustomerID"])


# Step 5: Feature engineering
def engineer_features(df):
    df["Income_Age_Ratio"] = df["Annual Income (k$)"] / df["Age"]
    df["Spending_Age_Ratio"] = df["Spending Score (1-100)"] / df["Age"]
    return df


# Step 6: Feature enrichment (simulated)
def enrich_features(df):
    df["Is_Senior"] = df["Age"] > 60
    df["Is_Young"] = df["Age"] < 25
    return df


# Step 7: Find optimal K using Elbow Method
def find_optimal_k(X, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    kl = KneeLocator(range(1, max_k + 1), wcss, curve="convex", direction="decreasing")
    optimal_k = kl.elbow

    plt.plot(range(1, max_k + 1), wcss, marker="o")
    plt.axvline(
        x=optimal_k, color="red", linestyle="--", label=f"Elbow at k={optimal_k}"
    )
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS (Inertia)")
    plt.legend()
    plt.show()

    return optimal_k


# 💾 Step 9: Export clustered data to CSV
def export_clusters(df, labels, filename="clustered_customers.csv"):
    df_with_labels = df.copy()
    df_with_labels["Cluster"] = labels
    df_with_labels.to_csv(filename, index=False)
    print(f"Clustered data saved to '{filename}'")


# Step 8: Final clustering and plot
def cluster_and_plot(X, df, optimal_k):
    kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    export_clusters(df, labels)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        df["Annual Income (k$)"], df["Spending Score (1-100)"], c=labels, cmap="viridis"
    )
    plt.scatter(
        kmeans.cluster_centers_[:, 1],
        kmeans.cluster_centers_[:, 2],
        color="red",
        marker="X",
        s=200,
    )
    plt.title(f"Customer Segments (k={optimal_k})")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.grid(True)
    plt.show()


# Pipeline execution
df = load_data()
df = handle_missing(df)
df = encode_features(df)
df = select_features(df)
df = engineer_features(df)
df = enrich_features(df)

X = df.values
optimal_k = find_optimal_k(X)
cluster_and_plot(X, df, optimal_k)
