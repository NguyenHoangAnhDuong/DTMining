"""
Step 01 — Data Cleaning
  1. Parse dates → cách ly hàng có ngày không hợp lệ ra CSV riêng (không tự đoán)
  2. Convert numeric → đổi '-' và chuỗi lỗi thành NaN
  3. Resolve duplicates:
       - Hàng trùng hoàn toàn → xóa, giữ hàng đầu
       - Cùng ngày nhưng giá trị mâu thuẫn → cách ly ra CSV riêng để kiểm tra thủ công
"""
import logging

import numpy as np
import pandas as pd

from .config import EVENTS, MACRO, MARKET, REPORTS_DIR

logger = logging.getLogger(__name__)

RAW_TARGETS = ["Tomorrow gold price", "Next week gold price", "Next Month gold price"]


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw = df["Date"].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False, errors="coerce")
    invalid = df["Date"].isna()
    if invalid.any():
        logger.warning("Quarantine %d invalid-date rows: %s", invalid.sum(), raw[invalid].tolist())
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        df.loc[invalid].to_csv(REPORTS_DIR / "quarantine_invalid_dates.csv", index=False)
    return df.loc[~invalid].copy()


def _convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in MARKET + MACRO + EVENTS + RAW_TARGETS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def _resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Hàng trùng hoàn toàn → xóa
    exact = df.duplicated(keep="first")
    if exact.any():
        logger.info("Removing %d exact duplicate rows", exact.sum())
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        df.loc[exact].to_csv(REPORTS_DIR / "exact_duplicates.csv", index=False)
    df = df.loc[~exact].copy()

    # Cùng ngày nhưng giá trị khác nhau → cách ly (không tự chọn hàng nào)
    conflicts = df["Date"].duplicated(keep=False)
    if conflicts.any():
        logger.warning("Quarantining %d rows with conflicting same-date values", conflicts.sum())
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        df.loc[conflicts].to_csv(REPORTS_DIR / "quarantine_conflicts.csv", index=False)
        df = df.loc[~conflicts].copy()

    return df.reset_index(drop=True)


def run(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 01 — Cleaning  |  input: %d rows × %d cols", *df.shape)
    df = _parse_dates(df)
    df = _convert_numeric(df)
    df = _resolve_duplicates(df)
    logger.info("Step 01 done  |  output: %d rows × %d cols", *df.shape)
    return df
