
import pandas as pd

from source.services.constants import (
    BASE_DATASET,
    DATA_REPORTS_DIR,
    RAW_DATA_DIR,
    SPLIT_DATA_DIR,
    TASKS,
)


def read_csv(path):
    """Đọc dữ liệu CSV nếu tệp tồn tại."""
    if not path.is_file():
        return pd.DataFrame()

    return pd.read_csv(path)


def load_raw_data():
    """Đọc dữ liệu gốc."""
    return read_csv(RAW_DATA_DIR / "dataset_root.csv")


def load_base_dataset():
    """Đọc dữ liệu sau tiền xử lý chung (Stage 1)."""
    return read_csv(BASE_DATASET)


def load_split_data(split="train", branch="base"):
    """Đọc tập Train / Validation / Test của một nhánh (base hoặc kmeans)."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Tập dữ liệu không hợp lệ: {split}")
    if branch not in TASKS:
        raise ValueError(f"Nhánh không hợp lệ: {branch}")

    return read_csv(SPLIT_DATA_DIR / f"{split}_{branch}.csv")


def load_preprocessing_report(report):
    """Đọc báo cáo tiền xử lý."""
    reports = {
        "clustering": "clustering_metrics.csv",
        "outlier": "outlier_summary.csv",
        "stationarity": "stationarity_report.csv",
    }

    if report not in reports:
        raise ValueError(f"Loại báo cáo không hợp lệ: {report}")

    return read_csv(DATA_REPORTS_DIR / reports[report])
