"""
رسومات استكشاف البيانات (EDA) لمشروع تقسيم العملاء.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.config import FIGURES_DIR


def plot_distributions(df: pd.DataFrame, columns: list, save: bool = True) -> None:
    """
    رسم histogram (مع KDE) لكل عمود رقمي في القائمة.

    Args:
        df (pd.DataFrame): البيانات.
        columns (list): أسماء الأعمدة المطلوب رسمها.
        save (bool): لو True، يحفظ كل رسمة في reports/figures/
    """
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)

    for col in columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True)
        plt.title(f"{col} Distribution")
        if save:
            safe_name = col.replace(" ", "_").replace("(", "").replace(")", "").replace("$", "")
            plt.savefig(os.path.join(FIGURES_DIR, f"dist_{safe_name}.png"), bbox_inches="tight")
        plt.show()


def plot_income_vs_spending(df: pd.DataFrame, save: bool = True) -> None:
    """
    رسم Scatter بين الدخل السنوي ودرجة الإنفاق.

    Args:
        df (pd.DataFrame): البيانات، لازم تحتوي على 'Annual Income (k$)'
                            و 'Spending Score (1-100)'.
        save (bool): لو True، يحفظ الرسمة.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)")
    plt.title("Annual Income (k$) VS Spending Score (1-100)")

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, "income_vs_spending.png"), bbox_inches="tight")
    plt.show()


def plot_clusters(df: pd.DataFrame, hue_col: str, title: str, save: bool = True) -> None:
    """
    رسم Scatter بين الدخل والإنفاق، ملوّن حسب الـ cluster.

    Args:
        df (pd.DataFrame): البيانات مع عمود الـ cluster labels.
        hue_col (str): اسم عمود التصنيف (مثلاً "KMeans_Labels").
        title (str): عنوان الرسمة.
        save (bool): لو True، يحفظ الرسمة.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue=hue_col,
        palette="tab10",
        s=100,
    )
    plt.title(title)

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        safe_name = hue_col.replace(" ", "_")
        plt.savefig(os.path.join(FIGURES_DIR, f"clusters_{safe_name}.png"), bbox_inches="tight")
    plt.show()
