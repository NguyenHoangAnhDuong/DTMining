"""
Lớp cache cho web app.

- Đọc file: @st.cache_data → mỗi file chỉ đọc 1 lần cho mọi phiên người dùng.
- Dự đoán : @st.cache_data theo (model, nhánh, h, ngày) → bấm lại cùng tham số trả ngay.
Các trang chỉ gọi hàm ở đây, không đọc file trực tiếp.
"""
import io
import zipfile

import pandas as pd
import streamlit as st

from source.services import data_loader, result_loader
from source.services.constants import MODELS, RESULTS_DIR
from source.services.forecast_service import forecast_at, min_anchor_index

TTL = None  # dữ liệu tĩnh; xoá cache bằng clear_all() khi chạy lại pipeline


def _dates(df, *cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


# ── Dữ liệu ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Đang nạp dữ liệu…", ttl=TTL)
def base_dataset() -> pd.DataFrame:
    df = data_loader.load_base_dataset()
    return _dates(df, "Date", "target_date_h1", "target_date_h7", "target_date_h30")


@st.cache_data(show_spinner=False, ttl=TTL)
def raw_dataset() -> pd.DataFrame:
    return data_loader.load_raw_data()


@st.cache_data(show_spinner=False, ttl=TTL)
def split_sizes() -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        df = data_loader.load_split_data(split, "base")
        if df.empty:
            continue
        d = pd.to_datetime(df["Date"])
        rows.append({"Tập": split, "Số phiên": len(df), "Từ": d.min().date(), "Đến": d.max().date()})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=TTL)
def preprocessing_report(name: str) -> pd.DataFrame:
    if name in {"clustering", "outlier", "stationarity"}:
        return data_loader.load_preprocessing_report(name)
    path = data_loader.DATA_REPORTS_DIR / f"{name}.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


@st.cache_data(show_spinner="Đang gán giai đoạn thị trường…", ttl=TTL)
def regime_history() -> pd.DataFrame:
    """Giai đoạn thị trường cho toàn bộ lịch sử (K-Means đã fit trên Train)."""
    from source.preprocessing_v3.step09_clustering import assign_regime

    df = assign_regime(base_dataset())
    return df[["Date", "Gold Price", "market_phase"]].dropna(subset=["market_phase"])


# ── Kết quả model ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=TTL)
def metrics(model: str | None = None) -> pd.DataFrame:
    if model is None:
        return result_loader.load_all_metrics()
    return result_loader.load_model_results(model, "metrics.csv")


@st.cache_data(show_spinner="Đang nạp kết quả dự đoán…", ttl=TTL)
def predictions(model: str) -> pd.DataFrame:
    return _dates(result_loader.load_model_results(model, "predictions.csv"), "Date")


@st.cache_data(show_spinner=False, ttl=TTL)
def feature_importance(model: str) -> pd.DataFrame:
    return result_loader.load_model_results(model, "feature_importance.csv")


@st.cache_data(show_spinner=False, ttl=TTL)
def metadata(model: str) -> dict:
    return result_loader.load_metadata(model)


@st.cache_data(show_spinner=False, ttl=TTL)
def forecast_summary(name: str) -> pd.DataFrame:
    return result_loader.load_forecast_summary(name)


@st.cache_data(show_spinner=False, ttl=TTL)
def kmeans_summary(name: str) -> pd.DataFrame:
    return result_loader.load_kmeans_summary(name)


@st.cache_data(show_spinner=False, ttl=TTL)
def report(name: str):
    text, folder = result_loader.load_report(name)
    return text, str(folder)


def available_models() -> list[str]:
    return [m for m in MODELS if not metrics(m).empty]


# ── Dự đoán ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang nạp mô hình…")
def _warm_model(model: str):
    """Import module model một lần cho cả server (xgboost, optuna, sklearn…)."""
    from source.services.forecast_service import _predictor

    return _predictor(model)


@st.cache_data(show_spinner=False, max_entries=512, ttl=TTL)
def forecast(model: str, branch: str, horizon: int, anchor_idx: int) -> dict:
    _warm_model(model)
    return forecast_at(base_dataset(), model, branch, horizon, anchor_idx)


def anchor_index(date) -> int:
    """Phiên giao dịch gần nhất ≤ ngày được chọn."""
    dates = base_dataset()["Date"]
    idx = int(dates.searchsorted(pd.Timestamp(date), side="right")) - 1
    return max(idx, min_anchor_index())


# ── Tải về ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Đang đóng gói kết quả…", ttl=TTL)
def results_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in RESULTS_DIR.rglob("*"):
            if path.is_file() and "models" not in path.parts:
                zf.write(path, path.relative_to(RESULTS_DIR.parent))
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def clear_all():
    st.cache_data.clear()
