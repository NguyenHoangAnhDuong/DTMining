
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
