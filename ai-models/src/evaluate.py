"""Đánh giá mô hình: metric cho bài toán 4 lớp và vẽ confusion matrix."""
import os

import matplotlib
matplotlib.use("Agg")  # vẽ ra file, không cần cửa sổ hiển thị
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay,
                             precision_recall_fscore_support)

LABEL_NAMES = ["Low", "Medium", "High", "Very High"]


def compute_metrics(y_true, y_pred):
    """Accuracy + Macro Precision/Recall/F1."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": p,
            "recall_macro": r,
            "f1_macro": f1}


def evaluate_train_test(pipe, X_train, y_train, X_test, y_test):
    """Metric trên cả train và test để so sánh mức độ overfitting."""
    return {"train": compute_metrics(y_train, pipe.predict(X_train)),
            "test": compute_metrics(y_test, pipe.predict(X_test))}


def save_confusion_matrix(y_true, y_pred, title, out_path):
    """Vẽ và lưu confusion matrix ra file ảnh, trả về ma trận."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm