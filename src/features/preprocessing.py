"""
تنظيف بيانات العملاء وتجهيزها قبل الـ clustering.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    تنظيف البيانات الخام:
    - حذف عمود CustomerID (مش مفيد في التحليل)
    - تحويل عمود Gender من نص إلى أرقام (0/1)

    Args:
        df (pd.DataFrame): البيانات الخام.

    Returns:
        pd.DataFrame: البيانات بعد التنظيف.
    """
    df = df.drop(columns=["CustomerID"], errors="ignore")

    # بعض نسخ الداتاسيت اسم العمود فيها "Genre" بدل "Gender"
    if "Genre" in df.columns and "Gender" not in df.columns:
        df = df.rename(columns={"Genre": "Gender"})

    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    return df


def scale_features(df: pd.DataFrame, columns: list) -> tuple:
    """
    توحيد مقياس الأعمدة (Standardization) قبل استخدامها في الـ clustering.

    ده مهم لأن خوارزميات زي KMeans و DBSCAN بتعتمد على المسافة بين
    النقط، فلازم كل الأعمدة تكون بنفس المقياس عشان عمود زي "الدخل"
    (بالآلاف) ميغلبش على عمود زي "درجة الإنفاق" (من 1 لـ 100).

    Args:
        df (pd.DataFrame): البيانات.
        columns (list): أسماء الأعمدة المطلوب توحيد مقياسها.

    Returns:
        tuple: (X_scaled كمصفوفة numpy, الـ scaler بعد التدريب عليه)
    """
    X = df[columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
