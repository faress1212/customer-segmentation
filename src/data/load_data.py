"""
تحميل بيانات عملاء المول (Mall Customers).

الداتاسيت من Kaggle:
https://www.kaggle.com/datasets/abdallahwagih/mall-customers-segmentation
"""

import os

import pandas as pd

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
