
import pandas as pd

from source.services.constants import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLIT_DATA_DIR,
    DATA_REPORTS_DIR,
)

PREPROCESSING_FILES = {
    "missing": "after_missing.csv",
    "outlier": "after_outlier.csv",
    "time_series": "after_time_series.csv",
    "stationarity": "after_stationarity.csv",
    "feature_engineering": "after_feature_engineering.csv",
    "feature_selection": "after_feature_selection.csv",
    "clustering": "after_clustering.csv",
}


def read_csv(path):
    """Đọc dữ liệu CSV nếu tệp tồn tại."""
    if not path.is_file():
        return pd.DataFrame()

    return pd.read_csv(path)


def load_raw_data():
    """Đọc dữ liệu gốc."""
    return read_csv(
        RAW_DATA_DIR / "dataset_root.csv"
    )


def load_processed_data(stage="feature_selection"):
    """Đọc dữ liệu theo bước tiền xử lý."""
    if stage not in PREPROCESSING_FILES:
        raise ValueError(
            f"Bước tiền xử lý không hợp lệ: {stage}"
        )

    return read_csv(
        PROCESSED_DATA_DIR / PREPROCESSING_FILES[stage]
    )


def load_split_data(split="train", scaled=False):
    """Đọc tập Train, Validation hoặc Test."""
    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"Tập dữ liệu không hợp lệ: {split}"
        )

    suffix = "_scaled" if scaled else ""

    return read_csv(
        SPLIT_DATA_DIR / f"{split}{suffix}.csv"
    )


def load_preprocessing_report(report):
    """Đọc báo cáo tiền xử lý."""
    reports = {
        "clustering": "clustering_report.csv",
        "feature_selection": "feature_selection_report.csv",
        "stationarity": "stationarity_report.csv",
    }

    if report not in reports:
        raise ValueError(
            f"Loại báo cáo không hợp lệ: {report}"
        )

    return read_csv(
        DATA_REPORTS_DIR / reports[report]
    )
