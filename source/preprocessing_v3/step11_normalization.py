"""
Step 11 — Normalization  (StandardScaler fit ONLY on Train)

Chỉ scale các cột feature đã được Feature Selection giữ lại.
Các cột sau KHÔNG scale (giữ nguyên giá trị gốc):
  - Giá gốc không dừng (Gold Price, Oil, v.v.) → SARIMAX cần trực tiếp
  - market_phase (chuỗi phân loại)
  - Cột target
  - Date

Sau scale: mean ≈ 0, std ≈ 1 trên tập Train;
  Val/Test có thể lệch nhẹ (phân phối khác Train) — đây là điều bình thường.

Lưu: feature_scaler.joblib trong dataset/processed_v3/models/
"""
import logging

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import MACRO, MARKET, MODELS_DIR

logger = logging.getLogger(__name__)

NO_SCALE = set(MARKET + MACRO) | {
    "Date", "market_phase",
    "Tomorrow gold price", "Next week gold price", "Next Month gold price",
    "target_price_h1", "target_price_h7", "target_price_h30",
    "target_logret_h1", "target_logret_h7", "target_logret_h30",
    "target_date_h1", "target_date_h7", "target_date_h30",
}


def _scale_cols(df: pd.DataFrame, selected: list[str]) -> list[str]:
    return [c for c in selected if c in df.columns and c not in NO_SCALE]


def run(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, selection_config: dict):
    logger.info("Step 11 — Normalization (StandardScaler fit on train)")

    selected  = selection_config.get("all_selected", [])
    feat_cols = _scale_cols(train, selected)
    logger.info("Scaling %d feature columns", len(feat_cols))

    scaler = StandardScaler()
    scaler.fit(train[feat_cols])

    train_sc, val_sc, test_sc = train.copy(), val.copy(), test.copy()
    train_sc[feat_cols] = scaler.transform(train[feat_cols])
    val_sc[feat_cols]   = scaler.transform(val[feat_cols])
    test_sc[feat_cols]  = scaler.transform(test[feat_cols])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.joblib")
    logger.info("Scaler saved → %s", MODELS_DIR / "feature_scaler.joblib")

    means = train_sc[feat_cols].mean()
    stds  = train_sc[feat_cols].std()
    logger.info("Post-scale train check  |  max|mean|=%.2e  std=[%.4f, %.4f]",
                means.abs().max(), stds.min(), stds.max())

    logger.info("Step 11 done")
    return train_sc, val_sc, test_sc, scaler
