"""
Lớp dữ liệu CHUNG cho mọi model — KHÔNG tạo feature, KHÔNG chuẩn hóa.

Chỉ cung cấp những thứ phải giống nhau giữa mọi model và cả 2 nhánh để so sánh công bằng:
  - Đọc dữ liệu 2 nhánh từ Stage 2 (base / kmeans) + nhãn giai đoạn để báo cáo theo giai đoạn
  - Target chuẩn: log-return h phiên tới (định nghĩa ở Stage 1, step06)
  - Purge: bỏ dòng có target nằm sau ranh giới Train / Val (tránh rò rỉ)
  - Quy đổi log-return → giá để đánh giá trên cùng thước đo

Feature và tham số là việc của từng model: source/model/<model>/features.py, train.py
"""
import numpy as np
import pandas as pd

from source.preprocessing_v3.config import KMEANS_SPLIT_SUFFIX, SPLIT_DIR, TRAIN_END, VAL_END
from source.preprocessing_v3.step09_clustering import REGIMES

GOLD_PX = "Gold Price"
GOLD_LR = "Gold Price_logret"
SPLITS  = ["train", "val", "test"]
BRANCH_NAMES = ("base", "kmeans")

# Cột nhãn giai đoạn do Stage 2 (K-Means) tạo — chỉ nhánh kmeans được dùng làm input model
REGIME_FEATURES = [f"kmeans_{r}" for r in REGIMES] + ["regime_id"]


def target_col(h: int) -> str:
    return f"target_logret_h{h}"


def target_price_col(h: int) -> str:
    return f"target_price_h{h}"


def target_date_col(h: int) -> str:
    return f"target_date_h{h}"


def read_branch(branch: str) -> pd.DataFrame:
    """Ghép train/val/test của một nhánh thành chuỗi liên tục (cột `split`) + cột `regime_label`."""
    if branch not in BRANCH_NAMES:
        raise ValueError(f"Nhánh không hợp lệ: {branch}")
    df = _read_split_files("base" if branch == "base" else KMEANS_SPLIT_SUFFIX)
    labels = _read_split_files(KMEANS_SPLIT_SUFFIX)[["Date", "market_phase"]]
    # regime_label chỉ để báo cáo kết quả theo giai đoạn, KHÔNG phải feature của nhánh base
    return df.merge(labels.rename(columns={"market_phase": "regime_label"}), on="Date", how="left")


def _read_split_files(suffix: str) -> pd.DataFrame:
    parts = [pd.read_csv(SPLIT_DIR / f"{sp}_{suffix}.csv", parse_dates=["Date"]).assign(split=sp)
             for sp in SPLITS]
    return pd.concat(parts, ignore_index=True).sort_values("Date").reset_index(drop=True)


def usable_mask(df: pd.DataFrame, h: int) -> pd.Series:
    """Dòng có target và không vi phạm ranh giới thời gian (purge)."""
    tdate = pd.to_datetime(df[target_date_col(h)])
    ok = df[target_col(h)].notna()
    ok &= ~((df["split"] == "train") & (tdate > pd.Timestamp(TRAIN_END)))
    ok &= ~((df["split"] == "val")   & (tdate > pd.Timestamp(VAL_END)))
    return ok


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {sp: df[df["split"] == sp].reset_index(drop=True) for sp in SPLITS}


def purge_fold(train: pd.DataFrame, fold: dict, h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia một fold walk-forward (trong Train) có purge theo ngày target."""
    d, td = train["Date"], pd.to_datetime(train[target_date_col(h)])
    tr_end, v_s, v_e = (pd.Timestamp(fold[k]) for k in ("train_end", "val_start", "val_end"))
    return (train[(d <= tr_end) & (td <= tr_end)],
            train[(d >= v_s) & (d <= v_e) & (td <= v_e)])


def to_price(df: pd.DataFrame, logret_pred: np.ndarray) -> np.ndarray:
    return df[GOLD_PX].to_numpy() * np.exp(logret_pred)


def prediction_frame(df: pd.DataFrame, lr_pred, model: str, branch: str, h: int, split: str) -> pd.DataFrame:
    """Định dạng predictions.csv chuẩn (docs/result_contract.md) — mọi model dùng chung."""
    return pd.DataFrame({
        "Date": df["Date"].to_numpy(), "model": model, "branch": branch, "horizon": h,
        "split": split, "regime_label": df["regime_label"].to_numpy(),
        "price_t": df[GOLD_PX].to_numpy(), "y_true": df[target_price_col(h)].to_numpy(),
        "y_pred": to_price(df, np.asarray(lr_pred)), "lr_true": df[target_col(h)].to_numpy(),
        "lr_pred": np.asarray(lr_pred),
    })
