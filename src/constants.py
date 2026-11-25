from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
ASSETS_DIR = ROOT_DIR / "assets"

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
    "task1": "With Clustering",
    "task2": "Without Clustering",
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
