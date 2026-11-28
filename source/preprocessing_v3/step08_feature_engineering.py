"""
Step 08 — Feature Engineering  (chạy trên toàn bộ dataset trước khi re-split)

Lý do chạy trên toàn bộ: lag/rolling cần lịch sử trước split boundary.
Không có data leakage vì:
  - Lag dùng shift(k), k >= 1  → chỉ nhìn vào quá khứ
  - Rolling dùng shift(1) trước khi rolling  → không dùng giá trị hiện tại

Nhóm feature:
  A. Log-return lags   : Gold Price_logret lags [1,2,3,5,7,10,20,30]
  B. Price lags        : Gold Price lags [1,2,3,5,7,10,20,30]
  C. Market lags       : Silver, Copper, Oil, DJI, S&P 500, USD index  lags [1,2,3,5]
  D. Rolling Gold      : mean, std, min, max của price và logret  windows [5,10,20,30,60]
  E. Cross-asset ratio : Gold/Silver, Gold/Oil, Gold/USD
  F. Calendar          : dayofweek, day, month, quarter, year
  G. Event lags        : {event}_lag1 cho 18 cột cờ sự kiện
  H. Macro diffs       : {macro}_diff + lags [1,2,3,5]
"""
import logging

import numpy as np
import pandas as pd

from .config import EVENTS, MACRO

logger = logging.getLogger(__name__)

GOLD_LAGS        = [1, 2, 3, 5, 7, 10, 20, 30]
MARKET_LAG_COLS  = ["Silver", "Copper", "Oil", "DJI", "S&P 500", "U.S dollar index"]
MARKET_LAGS      = [1, 2, 3, 5]
ROLLING_WINDOWS  = [5, 10, 20, 30, 60]

NON_FEATURE_COLS = {
    "Date", "market_phase",
    "Tomorrow gold price", "Next week gold price", "Next Month gold price",
    "target_price_h1", "target_price_h7", "target_price_h30",
    "target_logret_h1", "target_logret_h7", "target_logret_h30",
    "target_date_h1", "target_date_h7", "target_date_h30",
}


def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Gold Price_logret" in df.columns:
        for k in GOLD_LAGS:
            df[f"Gold Price_logret_lag{k}"] = df["Gold Price_logret"].shift(k)
    if "Gold Price" in df.columns:
        for k in GOLD_LAGS:
            df[f"Gold Price_lag{k}"] = df["Gold Price"].shift(k)
    for col in MARKET_LAG_COLS:
        if col in df.columns:
            for k in MARKET_LAGS:
                df[f"{col}_lag{k}"] = df[col].shift(k)
    return df


def _rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Gold Price_logret" not in df.columns or "Gold Price" not in df.columns:
        return df
    past_lr = df["Gold Price_logret"].shift(1)
    past_px = df["Gold Price"].shift(1)
    for w in ROLLING_WINDOWS:
        df[f"Gold_logret_roll_mean{w}"] = past_lr.rolling(w, min_periods=w).mean()
        df[f"Gold_logret_roll_std{w}"]  = past_lr.rolling(w, min_periods=w).std()
        df[f"Gold_price_roll_mean{w}"]  = past_px.rolling(w, min_periods=w).mean()
        df[f"Gold_price_roll_std{w}"]   = past_px.rolling(w, min_periods=w).std()
        df[f"Gold_price_roll_min{w}"]   = past_px.rolling(w, min_periods=w).min()
        df[f"Gold_price_roll_max{w}"]   = past_px.rolling(w, min_periods=w).max()
    return df


def _cross_asset_ratios(df: pd.DataFrame) -> pd.DataFrame:
    if "Gold Price" not in df.columns:
        return df
    for denom in ["Silver", "Oil", "U.S dollar index"]:
        if denom in df.columns:
            safe = df[denom].replace(0, np.nan)
            df[f"Gold_ratio_{denom}"] = df["Gold Price"] / safe
    return df


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["Date"])
    df["cal_dayofweek"] = dt.dt.dayofweek
    df["cal_day"]       = dt.dt.day
    df["cal_month"]     = dt.dt.month
    df["cal_quarter"]   = dt.dt.quarter
    df["cal_year"]      = dt.dt.year
    return df


def _event_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in EVENTS:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
    return df


def _macro_diff_lags(df: pd.DataFrame) -> pd.DataFrame:
    for col in MACRO:
        diff_col = col + "_diff"
        source   = df[diff_col] if diff_col in df.columns else None
        if source is None:
            continue
        for k in [1, 2, 3, 5]:
            df[f"{col}_diff_lag{k}"] = source.shift(k)
    return df


def _drop_warmup(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    before = len(df)
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    logger.info("Dropped %d warmup rows (NaN lag/rolling)  |  remaining: %d",
                before - len(df), len(df))
    return df


def run(df: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
    logger.info("Step 08 — Feature Engineering")
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)
    before_cols = set(df.columns)

    df = _lag_features(df)
    df = _rolling_features(df)
    df = _cross_asset_ratios(df)
    df = _calendar_features(df)
    df = _event_lag_features(df)
    df = _macro_diff_lags(df)

    new_cols = [c for c in df.columns if c not in before_cols]
    logger.info("Added %d feature columns", len(new_cols))

    if drop_warmup:
        df = _drop_warmup(df)

    logger.info("Step 08 done  |  total cols: %d", df.shape[1])
    return df
