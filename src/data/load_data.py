"""
تحميل بيانات عملاء المول (Mall Customers).

الداتاسيت من Kaggle:
https://www.kaggle.com/datasets/abdallahwagih/mall-customers-segmentation
"""

import os

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.config import RAW_DATA_PATH


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    تحميل ملف Mall_Customers.csv وتحويله لـ DataFrame.

    Args:
        path (str): مسار الملف (افتراضيًا data/raw/Mall_Customers.csv)

    Returns:
        pd.DataFrame: البيانات الخام.

    Raises:
        FileNotFoundError: لو الملف مش موجود.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"الملف مش موجود في '{path}'.\n"
            "حمّل الداتاسيت من Kaggle:\n"
            "https://www.kaggle.com/datasets/abdallahwagih/mall-customers-segmentation\n"
            "وحط ملف 'Mall_Customers.csv' جوه فولدر data/raw/"
        )
    return pd.read_csv(path)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    تنضيف بسيط للبيانات: إزالة التكرارات والقيم الناقصة.
    """
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def scale_features(df: pd.DataFrame, features: list):
    """
    تحجيم (Scaling) الأعمدة المحددة باستخدام StandardScaler.

    Args:
        df (pd.DataFrame): البيانات.
        features (list): أسماء الأعمدة المطلوب تحجيمها.

    Returns:
        tuple[np.ndarray, StandardScaler]: مصفوفة الأعمدة بعد التحجيم، وكائن الـ scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return X_scaled, scaler