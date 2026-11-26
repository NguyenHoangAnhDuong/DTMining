
import json
import pandas as pd

from source.services.constants import (
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
