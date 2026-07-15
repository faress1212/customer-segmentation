"""
Clustering utilities for the Customer Segmentation project: KMeans (with
elbow-method helper) and DBSCAN.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score


def elbow_method(X_scaled: np.ndarray, k_range: range = range(1, 11), save_path: str = None) -> list:
    """
    Compute and plot inertia (within-cluster sum of squares) for a range of
    K values, to help choose the optimal number of clusters (elbow method).

    Args:
        X_scaled (np.ndarray): Scaled feature matrix.
        k_range (range): Range of K values to test.
        save_path (str, optional): If provided, save the figure to this path.

    Returns:
        list: Inertia values for each K in k_range.
    """
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker="o")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return inertias


def run_kmeans(X_scaled: np.ndarray, n_clusters: int = 5, random_state: int = 42) -> tuple:
    """
    Fit a KMeans model and return the cluster labels and fitted model.

    Args:
        X_scaled (np.ndarray): Scaled feature matrix.
        n_clusters (int): Number of clusters.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (labels ndarray, fitted KMeans instance)
    """
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans


def run_dbscan(X_scaled: np.ndarray, eps: float = 0.3, min_samples: int = 5) -> tuple:
    """
    Fit a DBSCAN model and return the cluster labels and fitted model.

    Note: DBSCAN labels noise points as -1.

    Args:
        X_scaled (np.ndarray): Scaled feature matrix.
        eps (float): Maximum distance between two samples to be neighbors.
        min_samples (int): Minimum samples in a neighborhood to form a core point.

    Returns:
        tuple: (labels ndarray, fitted DBSCAN instance)
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    return labels, dbscan


def evaluate_clustering(X_scaled: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the silhouette score for a clustering result.

    Returns None if there are fewer than 2 distinct clusters (silhouette
    score is undefined in that case).

    Args:
        X_scaled (np.ndarray): Scaled feature matrix.
        labels (np.ndarray): Cluster labels.

    Returns:
        float or None: Silhouette score, or None if undefined.
    """
    n_unique_labels = len(set(labels) - {-1})
    if n_unique_labels < 2:
        return None
    return silhouette_score(X_scaled, labels)
