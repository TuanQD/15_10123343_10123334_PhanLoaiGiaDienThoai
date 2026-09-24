"""Khung huấn luyện dùng chung cho cả 4 mô hình + xuất model.joblib."""
import json
import os
import platform
import time
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from evaluate import evaluate_train_test
from preprocess import FEATURES, build_pipeline

# Cùng một cách CV cho cả 4 model
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(_BASE, "models")


def baseline(X_train, y_train, X_test, y_test):
    """Baseline: luôn đoán lớp phổ biến nhất. Dùng để so sánh."""
    pipe = build_pipeline(DummyClassifier(strategy="most_frequent"))
    pipe.fit(X_train, y_train)
    return evaluate_train_test(pipe, X_train, y_train, X_test, y_test)


def run_experiment(estimator, param_grid, X_train, y_train, X_test, y_test):
    """Chạy GridSearchCV cho một model. Dùng chung cho cả 4 model.

    Tên tham số trong param_grid phải có tiền tố 'model__',
    ví dụ {"model__C": [0.1, 1, 10]}.
    """
    pipe = build_pipeline(estimator)
    gs = GridSearchCV(pipe, param_grid, cv=CV, scoring="f1_macro",
                      n_jobs=-1, return_train_score=True)
    t0 = time.time()
    gs.fit(X_train, y_train)
    fit_seconds = time.time() - t0
    best = gs.best_estimator_
    return {
        "best_pipeline": best,
        "best_params": gs.best_params_,
        "cv_f1_macro": gs.best_score_,
        "cv_results": pd.DataFrame(gs.cv_results_),
        "metrics": evaluate_train_test(best, X_train, y_train, X_test, y_test),
        "fit_seconds": round(fit_seconds, 2),
    }


def export_model(pipeline, model_name, test_metrics,
                 out_dir=DEFAULT_MODEL_DIR, version="1.0.0"):
    """Lưu pipeline (tiền xử lý + model) thành model.joblib và metadata.json."""
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(out_dir, "model.joblib"), compress=3)
    meta = {
        "model_name": model_name,
        "model_version": version,
        "metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "trained_at": str(date.today()),
        "features": FEATURES,
        "libraries": {"python": platform.python_version(),
                      "scikit-learn": sklearn.__version__,
                      "numpy": np.__version__,
                      "pandas": pd.__version__},
    }
    with open(os.path.join(out_dir, "metadata.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta