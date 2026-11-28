"""
Step 03 — Missing Value Imputation
  Giá / chỉ số liên tục → nội suy tuyến tính theo thời gian (limit = 3 ngày),
      nếu còn NaN (khoảng trống dài hơn 3 ngày) thì ffill rồi bfill.
  Cột cờ sự kiện nhị phân (0/1) → ffill, đầu file còn trống thì fill 0.
  Giá dầu âm tháng 4/2020 là sự kiện thật — KHÔNG fill.
"""
import logging

import pandas as pd

from .config import EVENTS, MACRO, MARKET, REPORTS_DIR

logger = logging.getLogger(__name__)

RAW_TARGETS = ["Tomorrow gold price", "Next week gold price", "Next Month gold price"]
MAX_INTERP_GAP = 3

PRICE_COLS = MARKET + MACRO + RAW_TARGETS


def _impute_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("Date").sort_index()
    for col in PRICE_COLS:
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n == 0:
            continue
        logger.info("'%s': interpolating %d NaN (time, limit=%d)", col, n, MAX_INTERP_GAP)
        df[col] = (
            df[col]
            .interpolate(method="time", limit=MAX_INTERP_GAP, limit_direction="forward")
            .ffill()
            .bfill()
        )
    return df.reset_index()


def _impute_events(df: pd.DataFrame) -> pd.DataFrame:
    for col in EVENTS:
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n:
            logger.info("'%s': ffill %d NaN, fill remaining with 0", col, n)
            df[col] = df[col].ffill().fillna(0).astype(int)
    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 03 — Missing Value Imputation  |  total NaN before: %d",
                df.isnull().sum().sum())
    df = df.copy()
    df = _impute_prices(df)
    df = _impute_events(df)

    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]
    if not remaining.empty:
        logger.warning("NaN still remaining after imputation:\n%s", remaining.to_string())
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        remaining.to_csv(REPORTS_DIR / "remaining_missing.csv", header=["count"])
    else:
        logger.info("All NaN resolved.")

    logger.info("Step 03 done  |  total NaN after: %d", df.isnull().sum().sum())
    return df
