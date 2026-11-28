
import json
import pandas as pd

from source.services.constants import (
    CLUSTERING_RESULTS_DIR,
    FORECAST_RESULTS_DIR,
    MODELS,
)

RESULT_FILES = {
    "metrics.csv",
    "predictions.csv",
    "feature_importance.csv",
    "training_history.csv",
}


def load_model_results(model, filename):
    """Đọc kết quả của một mô hình."""
    if model not in MODELS:
        raise ValueError(
            f"Mô hình không hợp lệ: {model}"
        )

    if filename not in RESULT_FILES:
        raise ValueError(
            f"Tệp kết quả không hợp lệ: {filename}"
        )

    path = FORECAST_RESULTS_DIR / model / filename

    if not path.is_file():
        return pd.DataFrame()

    return pd.read_csv(path)


def load_all_metrics():
    """Tổng hợp chỉ số đánh giá của các mô hình."""
    frames = []

    for model in MODELS:
        df = load_model_results(model, "metrics.csv")

        if not df.empty:
            df["model"] = model
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_metadata(model):
    """Đọc thông tin cấu hình thí nghiệm."""
    if model not in MODELS:
        raise ValueError(
            f"Mô hình không hợp lệ: {model}"
        )

    path = FORECAST_RESULTS_DIR / model / "metadata.json"

    if not path.is_file():
        return {}

    with path.open(encoding="utf-8") as file:
        return json.load(file)


FORECAST_SUMMARIES = {
    "comparison": "result_forecast_comparison.csv",
    "by_regime": "result_forecast_by_regime.csv",
    "intervals": "result_forecast_intervals.csv",
    "intervals_by_regime": "result_forecast_intervals_by_regime.csv",
}

KMEANS_SUMMARIES = {
    "metrics": "result_k_mean_metrics.csv",
    "k_scan": "result_k_mean_k_scan.csv",
    "regimes": "result_k_mean_regimes.csv",
    "feature_comparison": "result_k_mean_feature_comparison.csv",
    "oos_usefulness": "result_k_mean_oos_usefulness.csv",
}

REPORTS = {
    "forecast": FORECAST_RESULTS_DIR / "result_forecast.md",
    "kmeans": CLUSTERING_RESULTS_DIR / "result_k_mean.md",
}


def _read(path):
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def load_forecast_summary(name):
    """Đọc bảng so sánh có/không phân lớp (sinh bởi report_forecast.py)."""
    return _read(FORECAST_RESULTS_DIR / FORECAST_SUMMARIES[name])


def load_kmeans_summary(name):
    """Đọc bảng kết quả phân cụm K-Means (sinh bởi report_kmeans.py)."""
    return _read(CLUSTERING_RESULTS_DIR / KMEANS_SUMMARIES[name])


def load_report(name):
    """Đọc báo cáo markdown, trả (nội dung, thư mục chứa báo cáo)."""
    path = REPORTS[name]
    if not path.is_file():
        return "", path.parent
    return path.read_text(encoding="utf-8"), path.parent
