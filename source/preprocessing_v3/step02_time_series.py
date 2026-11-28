"""
Step 02 — Time Series Structure
  1. Remove weekends (Thứ 7 / CN) — giá cuối tuần là copy giá Thứ Sáu, gây nhiễu ACF/PACF
     Thực hiện TRƯỚC khi impute missing để không lãng phí nội suy trên hàng sẽ bị xóa
  2. Sort theo Date, reset index
  3. Log các khoảng trống ngày lễ >= 4 ngày lịch (không fill — thị trường thực sự đóng cửa)
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def run(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 02 — Time Series Structure  |  input: %d rows", len(df))
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    before = len(df)
    df = df[df["Date"].dt.dayofweek < 5].copy()   # 0=Mon … 4=Fri
    logger.info("Removed %d weekend rows  |  remaining: %d", before - len(df), len(df))

    df = df.sort_values("Date").reset_index(drop=True)

    # Log holiday gaps (khoảng trống >= 4 ngày giữa 2 ngày giao dịch liên tiếp)
    diffs = df["Date"].diff().dropna()
    gaps  = diffs[diffs.dt.days >= 4]
    if gaps.empty:
        logger.info("No holiday gaps >= 4 calendar days found.")
    else:
        logger.info("%d holiday gaps >= 4 days:", len(gaps))
        for idx in gaps.index:
            b = df.loc[idx - 1, "Date"]
            a = df.loc[idx,     "Date"]
            logger.info("  %s (%s) → %s (%s)  [%d days]",
                        b.date(), b.day_name(), a.date(), a.day_name(), diffs.loc[idx].days)

    n_dup = df["Date"].duplicated().sum()
    if n_dup:
        logger.warning("Still %d duplicate dates remaining!", n_dup)
    else:
        logger.info("No duplicate dates.")

    logger.info("Step 02 done  |  range: %s → %s", df["Date"].min().date(), df["Date"].max().date())
    return df
