"""
تجميع العملاء في فئات (Clustering) باستخدام موديلات جاهزة من scikit-learn:
- KMeans: بيقسم العملاء لعدد محدد (K) من المجموعات.
- DBSCAN: بيكتشف المجموعات تلقائيًا من كثافة النقط، وبيقدر يكتشف
  الـ outliers (نقط "شاذة" مش تابعة لأي مجموعة، بتاخد label = -1).
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from src.utils.config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE


def elbow_method(X_scaled: np.ndarray, k_range: range = range(1, 11), save: bool = True) -> list:
    """
    حساب ورسم الـ inertia لعدة قيم K، عشان نساعد نختار أفضل عدد
    مجموعات (طريقة الـ Elbow / الكوع).

    الفكرة: كل ما K يزيد، الـ inertia (مجموع المسافات جوه كل مجموعة)
    بيقل. بندور على النقطة اللي بعدها التحسن بيبقى بطيء جدًا (شكلها
    زي كوع في الرسمة)، وده يكون أنسب K.

    Args:
        X_scaled (np.ndarray): البيانات بعد التوحيد (scaling).
        k_range (range): مدى قيم K المراد تجربتها.
        save (bool): لو True، يحفظ الرسمة.

    Returns:
        list: قيم الـ inertia لكل K.
    """
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker="o")
    plt.title("طريقة الكوع (Elbow Method) لاختيار أفضل K")
    plt.xlabel("عدد المجموعات (K)")
    plt.ylabel("Inertia")

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, "elbow_method.png"), bbox_inches="tight")
    plt.show()

    return inertias


def run_kmeans(X_scaled: np.ndarray, n_clusters: int = 5) -> tuple:
    """
    تدريب موديل KMeans وإرجاع الـ labels والموديل.

    Args:
        X_scaled (np.ndarray): البيانات بعد التوحيد.
        n_clusters (int): عدد المجموعات المطلوبة.

    Returns:
        tuple: (labels, موديل KMeans المدرب)
    """
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans


def run_dbscan(X_scaled: np.ndarray, eps: float = 0.3, min_samples: int = 5) -> tuple:
    """
    تدريب موديل DBSCAN وإرجاع الـ labels والموديل.

    ملحوظة: DBSCAN بيدي label = -1 للنقط اللي يعتبرها "ضوضاء" (noise)
    ومش تابعة لأي مجموعة.

    Args:
        X_scaled (np.ndarray): البيانات بعد التوحيد.
        eps (float): أقصى مسافة بين نقطتين عشان يعتبرهم جيران.
        min_samples (int): أقل عدد نقط مطلوب عشان تتكوّن مجموعة.

    Returns:
        tuple: (labels, موديل DBSCAN المدرب)
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    return labels, dbscan


def evaluate_clustering(X_scaled: np.ndarray, labels: np.ndarray):
    """
    حساب الـ silhouette score لتقييم جودة التقسيم.

    القيمة بتتراوح من -1 لـ 1: كل ما أقرب لـ 1 يبقى التقسيم أوضح
    (المجموعات متباعدة عن بعض ومتماسكة جوه نفسها).

    Args:
        X_scaled (np.ndarray): البيانات بعد التوحيد.
        labels (np.ndarray): تصنيفات المجموعات.

    Returns:
        float أو None: الـ silhouette score، أو None لو أقل من مجموعتين.
    """
    n_unique_labels = len(set(labels) - {-1})
    if n_unique_labels < 2:
        return None
    return silhouette_score(X_scaled, labels)


def save_model(model, filename: str) -> None:
    """
    حفظ موديل مدرب كملف .pkl في فولدر models/

    Args:
        model: الموديل المدرب.
        filename (str): اسم الملف، مثلاً "kmeans_model.pkl"
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"تم حفظ الموديل في: {path}")
