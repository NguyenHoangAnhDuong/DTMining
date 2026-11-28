
from source.preprocessing_v3.config import (
    BASE_DATASET,
    BRANCHES,
    FIGURES_DIR,
    FORECAST_RESULTS,
    HORIZONS,
    KMEANS_RESULTS,
    OUT_DIR as PROCESSED_DATA_DIR,
    RAW_DATA,
    REPORTS_DIR as DATA_REPORTS_DIR,
    RESULTS_DIR,
    ROOT_DIR,
    SPLIT_DIR as SPLIT_DATA_DIR,
)

# Mọi đường dẫn lấy từ source/preprocessing_v3/config.py (một nguồn duy nhất)
DATA_DIR = ROOT_DIR / "dataset"
ASSETS_DIR = ROOT_DIR / "assets"
RAW_DATA_DIR = RAW_DATA.parent

FORECAST_RESULTS_DIR = FORECAST_RESULTS
CLUSTERING_RESULTS_DIR = KMEANS_RESULTS

APP_NAME = "Gold Intelligence"

# Model đã có kết quả nằm trong results/forecasting/<model>/
MODELS = [
    "xgboost",
]

MODEL_NAMES = {
    "xgboost": "XGBoost",
}

# Hai nhánh thí nghiệm: có / không phân lớp K-Means
TASKS = BRANCHES

CLUSTERING_ALGORITHMS = [
    "kmeans",
]

METRICS = [
    "MAE",
    "RMSE",
    "MAPE",
    "R2",
    "MASE",
    "DA",
]

# Module chứa `predict_latest(history, branch, horizon)` của từng model (dùng cho trang Dự đoán)
MODEL_MODULES = {
    "xgboost": "source.model.xgboost_model.train",
}

BRANCH_LABELS = {
    "base": "Không phân lớp",
    "kmeans": "Có phân lớp K-Means",
}

SPLIT_LABELS = {
    "cv": "Walk-forward (2004–2017)",
    "val": "Validation (2018–2019)",
    "test": "Test (2020–2022)",
}

# Giai đoạn thị trường (thứ tự theo REGIMES trong step09_clustering)
REGIMES = ["Bear", "Bull", "StrongBull", "Volatile"]

REGIME_LABELS = {
    "Bear": "Giảm",
    "Bull": "Tăng",
    "StrongBull": "Tăng mạnh",
    "Volatile": "Biến động mạnh",
}

REGIME_COLORS = {
    "Bear": "#B23A3A",
    "Bull": "#4A7C3A",
    "StrongBull": "#1F6F78",
    "Volatile": "#7A4E9A",
}

# Màu dùng chung cho biểu đồ (nền sáng, màu trầm)
COLORS = {
    "gold": "#9A6B12",
    "actual": "#1C1B19",
    "base": "#2F5D8C",
    "kmeans": "#C0562B",
    "naive": "#8A8680",
    "good": "#3E7B4F",
    "bad": "#B23A3A",
    "rule": "#8A8680",
    "grid": "#E9E6DE",
    "axis": "#B9B4A8",
    "text": "#3A3833",
    "muted": "#6B675F",
    "bg": "#FFFFFF",
}
