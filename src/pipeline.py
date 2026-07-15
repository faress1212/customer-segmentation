"""
End-to-end pipeline function for the Customer Segmentation project.
Chains data loading, preprocessing, scaling, and clustering (KMeans +
DBSCAN) into a single callable, driven by config.yaml.
"""

from src.data.load_data import load_data, preprocess, scale_features
from src.models.train_models import evaluate_clustering, run_dbscan, run_kmeans
from src.utils.config import load_config


def run_pipeline(config: dict = None) -> dict:
    """
    Run the full customer segmentation pipeline: load data, preprocess,
    scale features, and run both KMeans and DBSCAN clustering.

    Args:
        config (dict, optional): Parsed config. Loads config/config.yaml
                                  if not provided.

    Returns:
        dict: {
            "df": dataframe with cluster label columns added,
            "kmeans_labels": ndarray,
            "kmeans_silhouette": float,
            "dbscan_labels": ndarray,
            "dbscan_silhouette": float or None,
        }
    """
    config = config or load_config()

    # 1. Load & preprocess
    df = load_data(config["data"]["raw_path"])
    df = preprocess(df)

    # 2. Scale clustering features
    feature_cols = config["features"]["clustering_features"]
    X_scaled, scaler = scale_features(df, feature_cols)

    # 3. KMeans
    kmeans_cfg = config["clustering"]["kmeans"]
    kmeans_labels, kmeans_model = run_kmeans(
        X_scaled,
        n_clusters=kmeans_cfg["n_clusters"],
        random_state=kmeans_cfg["random_state"],
    )
    df["KMeans_Labels"] = kmeans_labels
    kmeans_silhouette = evaluate_clustering(X_scaled, kmeans_labels)

    # 4. DBSCAN
    dbscan_cfg = config["clustering"]["dbscan"]
    dbscan_labels, dbscan_model = run_dbscan(
        X_scaled,
        eps=dbscan_cfg["eps"],
        min_samples=dbscan_cfg["min_samples"],
    )
    df["DBSCAN_Labels"] = dbscan_labels
    dbscan_silhouette = evaluate_clustering(X_scaled, dbscan_labels)

    return {
        "df": df,
        "kmeans_labels": kmeans_labels,
        "kmeans_silhouette": kmeans_silhouette,
        "dbscan_labels": dbscan_labels,
        "dbscan_silhouette": dbscan_silhouette,
    }
