"""
نقطة تشغيل مشروع تقسيم العملاء (Customer Segmentation)، خطوة بخطوة.

يتطلب وجود data/raw/Mall_Customers.csv (شوف data/README.md لطريقة التحميل).

تشغيل:
    python main.py
"""

from src.data.load_data import load_data
from src.features.preprocessing import preprocess, scale_features
from src.models.clustering import (
    elbow_method,
    evaluate_clustering,
    run_dbscan,
    run_kmeans,
    save_model,
)
from src.utils.config import CLUSTERING_FEATURES, N_CLUSTERS
from src.visualization.eda_plots import (
    plot_clusters,
    plot_distributions,
    plot_income_vs_spending,
)


def main():
    # 1. تحميل وتنظيف البيانات
    df = load_data()
    df = preprocess(df)
    print(df.head())

    # 2. رسومات استكشافية (EDA)
    plot_distributions(df, ["Age", "Annual Income (k$)", "Spending Score (1-100)"])
    plot_income_vs_spending(df)

    # 3. توحيد مقياس الأعمدة المستخدمة في الـ clustering
    X_scaled, scaler = scale_features(df, CLUSTERING_FEATURES)

    # 4. طريقة الكوع لمساعدتنا نختار أفضل K (تطبع قيم الـ inertia وترسمها)
    inertias = elbow_method(X_scaled)
    print("\nInertia for each K:", inertias)

    # 5. تجميع بـ KMeans
    kmeans_labels, kmeans_model = run_kmeans(X_scaled, n_clusters=N_CLUSTERS)
    df["KMeans_Labels"] = kmeans_labels
    kmeans_score = evaluate_clustering(X_scaled, kmeans_labels)
    print(f"\nKMeans silhouette score: {kmeans_score:.4f}")
    plot_clusters(df, "KMeans_Labels", "Divide Customers By KMeans")

    # 6. تجميع بـ DBSCAN
    dbscan_labels, dbscan_model = run_dbscan(X_scaled, eps=0.3, min_samples=5)
    df["DBSCAN_Labels"] = dbscan_labels
    dbscan_score = evaluate_clustering(X_scaled, dbscan_labels)
    n_noise = (dbscan_labels == -1).sum()
    print(f"DBSCAN silhouette score: {dbscan_score}")
    print(f"Noise in DBSCAN: {n_noise} / {len(dbscan_labels)}")
    plot_clusters(df, "DBSCAN_Labels", "Customer Divided by DBSCAN")

    # 7. حفظ موديل KMeans (الأنسب هنا لأن عدد المجموعات معروف ومحدد)
    save_model(kmeans_model, "kmeans_model.pkl")


if __name__ == "__main__":
    main()
