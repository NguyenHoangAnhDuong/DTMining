
from pathlib import Path

# Thư mục gốc của dự án
ROOT_DIR = Path(__file__).resolve().parents[2]

# Các thư mục chính
DATA_DIR = ROOT_DIR / "dataset"
RESULTS_DIR = ROOT_DIR / "result"
ASSETS_DIR = ROOT_DIR / "assets"
FIGURES_DIR = ROOT_DIR / "figures"

# Các thư mục dữ liệu
RAW_DATA_DIR = DATA_DIR / "data_root"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATA_REPORTS_DIR = DATA_DIR / "reports"
SPLIT_DATA_DIR = PROCESSED_DATA_DIR / "split"

# Các thư mục kết quả
FORECAST_RESULTS_DIR = RESULTS_DIR / "forecasting"
CLUSTERING_RESULTS_DIR = RESULTS_DIR / "clustering"
EDA_RESULTS_DIR = RESULTS_DIR / "eda"
PREPROCESSING_RESULTS_DIR = RESULTS_DIR / "preprocessing"

APP_NAME = "Gold Intelligence"

MODELS = [
    "sarimax",
    "xgboost",
    "lightgbm",
    "prophet",
    "lstm",
]

MODEL_NAMES = {
    "sarimax": "SARIMAX",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "prophet": "Prophet",
    "lstm": "LSTM",
}

TASKS = {
    "task1": "Có sử dụng kết quả phân cụm",
    "task2": "Không sử dụng kết quả phân cụm",
}

HORIZONS = [1, 7, 30]

CLUSTERING_ALGORITHMS = [
    "kmeans",
    "gmm",
]

METRICS = [
    "MAE",
    "RMSE",
    "MAPE",
    "R2",
    "MASE",
    "Directional_Accuracy",
]
