"""
إعدادات ومسارات المشروع، كلها في مكان واحد عشان سهل نغيرها.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Mall_Customers.csv")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

# الأعمدة المستخدمة في عملية الـ clustering
CLUSTERING_FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]

RANDOM_STATE = 42
N_CLUSTERS = 5

def load_config() -> dict:
    """
    ترجع كل الإعدادات في شكل dictionary عشان تستخدم بسهولة في النوتبوك.
    """
    return {
        "data": {
            "raw_path": RAW_DATA_PATH,
        },
        "models_dir": MODELS_DIR,
        "figures_dir": FIGURES_DIR,
        "clustering_features": CLUSTERING_FEATURES,
        "random_state": RANDOM_STATE,
        "n_clusters": N_CLUSTERS,
    }