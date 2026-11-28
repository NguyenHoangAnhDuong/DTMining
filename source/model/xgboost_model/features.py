"""
Feature RIÊNG của XGBoost (model cây dạng bảng) — tính trong bộ nhớ lúc train / dự đoán, không lưu ra file.

Vì sao chọn như vậy (đặc thù của model cây):
  - Chỉ dùng đại lượng dừng (log-return, sai phân, rolling của return, cờ sự kiện, lịch):
    cây không ngoại suy được khi giá Test vượt đỉnh Train → không dùng giá tuyệt đối.
  - Không cần chuẩn hóa (cây bất biến với phép biến đổi đơn điệu).
  - Nhánh kmeans: thêm REGIME_FEATURES (nhãn giai đoạn) — khác biệt DUY NHẤT với nhánh base.

Dòng hợp lệ xác định theo feature GỐC → 2 nhánh có cùng các dòng.
"""
import pandas as pd

from source.model.common.data import (
    GOLD_LR, GOLD_PX, REGIME_FEATURES, read_branch, split_frames, target_col, usable_mask,
)
from source.preprocessing_v3.config import EVENTS

GOLD_LAGS  = [1, 2, 3, 5, 10, 20]
WINDOWS    = [5, 10, 20, 60]
MA_WINDOWS = [50, 200]
SUM_ASSETS = ["Silver_logret", "U.S dollar index_logret", "S&P 500_logret", "U.S Treasury_logret"]


def _event_cols(df: pd.DataFrame) -> list[str]:
    names = {e.strip() for e in EVENTS}
    return [c for c in df.columns if c.strip() in names]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Trả về (df có thêm cột f_*, danh sách cột feature gốc). df phải liên tục theo thời gian."""
    df = df.sort_values("Date").reset_index(drop=True)
    lr, px = df[GOLD_LR], df[GOLD_PX]
    f = {}

    # Biến động trong ngày của mọi tài sản / vĩ mô (đã dừng từ Stage 1)
    for c in df.columns:
        if c.endswith("_logret") or c.endswith("_diff"):
            f[c] = df[c]
    # Trễ và rolling của log-return vàng (gồm cả ngày t — đã biết tại thời điểm dự đoán)
    for k in GOLD_LAGS:
        f[f"gold_lr_lag{k}"] = lr.shift(k)
    for w in WINDOWS:
        f[f"gold_lr_mean{w}"] = lr.rolling(w).mean()
        f[f"gold_lr_std{w}"]  = lr.rolling(w).std()
    for w in MA_WINDOWS:
        f[f"gold_ma{w}_gap"] = px / px.rolling(w).mean() - 1
    # Return tích lũy 5 phiên của tài sản liên quan
    for c in SUM_ASSETS:
        if c in df.columns:
            f[f"{c}_sum5"] = df[c].rolling(5).sum()
    # Sự kiện: cờ ngày t + số sự kiện trong 5 phiên gần nhất
    for c in _event_cols(df):
        key = c.strip()
        f[f"evt_{key}"]      = df[c]
        f[f"evt_{key}_cnt5"] = df[c].rolling(5).sum()
    # Lịch
    dt = pd.to_datetime(df["Date"])
    f["cal_dayofweek"] = dt.dt.dayofweek
    f["cal_month"]     = dt.dt.month

    feats = pd.DataFrame(f, index=df.index).add_prefix("f_")
    return pd.concat([df, feats], axis=1), list(feats.columns)


def feature_columns(base_feats: list[str], branch: str) -> list[str]:
    return base_feats + (REGIME_FEATURES if branch == "kmeans" else [])


def make_dataset(branch: str, h: int) -> tuple[dict[str, pd.DataFrame], list[str], str]:
    """({split: df}, feature_cols, target_col) cho một nhánh và một khoảng dự đoán."""
    df, base_feats = build_features(read_branch(branch))
    ok = df[base_feats].notna().all(axis=1) & usable_mask(df, h)
    return split_frames(df[ok]), feature_columns(base_feats, branch), target_col(h)
