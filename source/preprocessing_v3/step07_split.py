"""
Step 07 — Temporal Split + Walk-Forward CV

Split theo mốc ngày cứng:
  Train      : đầu dataset → 2017-12-31  (~80%)
  Validation : 2018-01-01  → 2019-12-31  (~9%)
  Test       : 2020-01-01  → cuối dataset (~11%)

Walk-forward expanding-window (trên Train):
  Mỗi fold mở rộng thêm 1 năm, val window = 1 năm.
  Lưu ra JSON để các model dùng lại khi walk-forward CV.

Lưu ra: train_base.csv, val_base.csv, test_base.csv
  ("base" = chưa có feature engineering, chỉ là split từ base_dataset)
"""
import json
import logging

import pandas as pd

from .config import (
    SPLIT_DIR,
    TEST_START, TRAIN_END, VAL_END, VAL_START,
    WF_MIN_TRAIN_YEARS, WF_STEP_YEARS, WF_VAL_YEARS,
)

logger = logging.getLogger(__name__)


def temporal_split(df: pd.DataFrame):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    train = df[df["Date"] <= TRAIN_END].reset_index(drop=True)
    val   = df[(df["Date"] >= VAL_START) & (df["Date"] <= VAL_END)].reset_index(drop=True)
    test  = df[df["Date"] >= TEST_START].reset_index(drop=True)
    total = len(train) + len(val) + len(test)
    logger.info(
        "Split — Train: %d (%.1f%%)  Val: %d (%.1f%%)  Test: %d (%.1f%%)",
        len(train), 100 * len(train) / total,
        len(val),   100 * len(val)   / total,
        len(test),  100 * len(test)  / total,
    )
    logger.info(
        "  Train: %s → %s | Val: %s → %s | Test: %s → %s",
        train["Date"].min().date(), train["Date"].max().date(),
        val["Date"].min().date(),   val["Date"].max().date(),
        test["Date"].min().date(),  test["Date"].max().date(),
    )
    return train, val, test


def walk_forward_splits(train_df: pd.DataFrame) -> list[dict]:
    df = train_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    min_yr = df["Date"].dt.year.min()
    max_yr = df["Date"].dt.year.max()
    first_end = min_yr + WF_MIN_TRAIN_YEARS - 1
    folds = []
    fold_id = 1
    for end_yr in range(first_end, max_yr - WF_VAL_YEARS + 1, WF_STEP_YEARS):
        val_s = end_yr + 1
        val_e = val_s + WF_VAL_YEARS - 1
        if val_e > max_yr:
            break
        f_tr = df[df["Date"].dt.year <= end_yr]
        f_vl = df[(df["Date"].dt.year >= val_s) & (df["Date"].dt.year <= val_e)]
        if len(f_tr) == 0 or len(f_vl) == 0:
            continue
        folds.append({
            "fold":        fold_id,
            "train_start": str(f_tr["Date"].min().date()),
            "train_end":   str(f_tr["Date"].max().date()),
            "val_start":   str(f_vl["Date"].min().date()),
            "val_end":     str(f_vl["Date"].max().date()),
            "train_rows":  len(f_tr),
            "val_rows":    len(f_vl),
        })
        fold_id += 1
    logger.info("Generated %d walk-forward folds", len(folds))
    return folds


def run(df: pd.DataFrame):
    logger.info("Step 07 — Temporal Split")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    train, val, test = temporal_split(df)
    wf_folds = walk_forward_splits(train)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(SPLIT_DIR / "train_base.csv", index=False)
    val.to_csv(SPLIT_DIR   / "val_base.csv",   index=False)
    test.to_csv(SPLIT_DIR  / "test_base.csv",  index=False)

    wf_path = SPLIT_DIR / "walk_forward_splits.json"
    with open(wf_path, "w", encoding="utf-8") as fh:
        json.dump(wf_folds, fh, indent=2, ensure_ascii=False)
    logger.info("Step 07 done  |  walk-forward JSON → %s", wf_path)

    return train, val, test, wf_folds
