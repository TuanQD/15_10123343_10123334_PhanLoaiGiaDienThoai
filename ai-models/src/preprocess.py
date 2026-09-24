"""Tiền xử lý dữ liệu dùng chung cho cả 4 mô hình.

Quy tắc:
- Danh sách đặc trưng (FEATURES) phải khớp schema.json.
- Scaler/Imputer nằm TRONG pipeline, nên chỉ fit trên tập train (không leakage).
- Chỉ dùng transformer có sẵn của sklearn để joblib nạp lại dễ dàng.
"""
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "price_range"
RANDOM_STATE = 42
TEST_SIZE = 0.2

BINARY_FEATURES = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]
NUMERIC_FEATURES = ["battery_power", "clock_speed", "fc", "int_memory", "m_dep",
                    "mobile_wt", "n_cores", "pc", "px_height", "px_width",
                    "ram", "sc_h", "sc_w", "talk_time"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES  # thứ tự chuẩn, dùng ở mọi nơi

# Đường dẫn tính từ vị trí file này, chạy ở đâu cũng đúng
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(_BASE, "data", "train.csv")


def load_data(path=DEFAULT_DATA_PATH):
    """Đọc CSV, trả về (X, y) với X theo đúng thứ tự FEATURES."""
    df = pd.read_csv(path)
    return df[FEATURES], df[TARGET]


def split_data(X, y):
    """Chia train/test MỘT LẦN, dùng chung cho cả 4 model."""
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)


def build_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC_FEATURES),
        ("bin", SimpleImputer(strategy="most_frequent"), BINARY_FEATURES),
    ])


def build_pipeline(model):
    """Pipeline hoàn chỉnh: tiền xử lý + model."""
    return Pipeline([("prep", build_preprocessor()), ("model", model)])