"""
Step 10 — Feature Selection  (fit ONLY on Train, không leakage)

Quy trình:
  1. Tính Mutual Information (MI) của từng feature với target_price_h1
  2. Lọc tương quan cao: |corr| > 0.95 → giữ cột có MI cao hơn, bỏ cột còn lại
  3. Tính lại MI sau lọc tương quan
  4. Loại feature có MI = 0

Lưu:
  - feature_selection_report.csv : chi tiết từng feature (mi, status, reason)
  - feature_selection.json       : config các subset top-20/40/60/80 + all_selected

Giá gốc không dừng (RAW_PRICE_COLS) được GIỮ trong output nhưng KHÔNG tham gia lọc feature.
Lý do: SARIMAX cần cột giá gốc; loại chúng khỏi ML feature pool là đúng vì chúng không dừng.
"""
import json
import logging

import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from .config import MACRO, MARKET, REPORTS_DIR

logger = logging.getLogger(__name__)

CORR_THRESHOLD = 0.95
TARGET_COL     = "target_price_h1"

# Cột không được tham gia feature selection
RAW_PRICE_COLS = set(MARKET + MACRO)

NO_FEATURE = RAW_PRICE_COLS | {
    "Date", "market_phase",
    "Tomorrow gold price", "Next week gold price", "Next Month gold price",
    "target_price_h1", "target_price_h7", "target_price_h30",
    "target_logret_h1", "target_logret_h7", "target_logret_h30",
    "target_date_h1", "target_date_h7", "target_date_h30",
}


def _get_candidates(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NO_FEATURE]


def _compute_mi(df: pd.DataFrame, features: list[str], target: pd.Series) -> pd.Series:
    X      = df[features].fillna(df[features].median())
    scores = mutual_info_regression(X.values, target.values, random_state=42)
    return pd.Series(scores, index=features).sort_values(ascending=False)


def _corr_filter(df: pd.DataFrame, features: list[str], mi: pd.Series):
    corr     = df[features].corr().abs()
    removed: set[str] = set()
    log_rows = []
    for i, col_a in enumerate(features):
        if col_a in removed:
            continue
        for col_b in features[i + 1:]:
            if col_b in removed:
                continue
            if corr.loc[col_a, col_b] > CORR_THRESHOLD:
                drop = col_b if mi.get(col_a, 0) >= mi.get(col_b, 0) else col_a
                keep = col_a if drop == col_b else col_b
                removed.add(drop)
                log_rows.append({
                    "dropped"   : drop, "kept": keep,
                    "corr"      : round(float(corr.loc[col_a, col_b]), 4),
                    "mi_dropped": round(float(mi.get(drop, 0)), 6),
                    "mi_kept"   : round(float(mi.get(keep, 0)), 6),
                })
    kept = [c for c in features if c not in removed]
    return kept, log_rows


def run(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    logger.info("Step 10 — Feature Selection (fit on train only)")

    cands       = _get_candidates(train)
    train_valid = train.dropna(subset=[TARGET_COL])
    target      = train_valid[TARGET_COL]

    logger.info("Candidates: %d  |  Train rows with valid target: %d", len(cands), len(train_valid))

    mi_first = _compute_mi(train_valid, cands, target)
    kept_after_corr, corr_log = _corr_filter(train_valid, cands, mi_first)
    logger.info("After corr filter (>%.2f): %d / %d kept (%d dropped)",
                CORR_THRESHOLD, len(kept_after_corr), len(cands),
                len(cands) - len(kept_after_corr))

    mi_final = _compute_mi(train_valid, kept_after_corr, target)
    selected = mi_final[mi_final > 0].index.tolist()
    zero_mi  = mi_final[mi_final == 0].index.tolist()
    logger.info("After zero-MI filter: %d selected  |  %d dropped", len(selected), len(zero_mi))

    # ── Báo cáo ───────────────────────────────────────────────────────────
    removed_by_corr = {r["dropped"] for r in corr_log}
    rows = []
    for c in cands:
        if c in removed_by_corr:
            rows.append({"feature": c, "mi": round(float(mi_first.get(c, 0)), 6),
                         "status": "removed", "reason": f"corr>{CORR_THRESHOLD}"})
        elif c in zero_mi:
            rows.append({"feature": c, "mi": 0.0, "status": "removed", "reason": "zero_mi"})
        else:
            rows.append({"feature": c, "mi": round(float(mi_final.get(c, 0)), 6),
                         "status": "kept", "reason": ""})

    report = (pd.DataFrame(rows)
              .sort_values(["status", "mi"], ascending=[True, False])
              .reset_index(drop=True))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORTS_DIR / "feature_selection_report.csv", index=False)

    config = {
        "all_selected": selected,
        "top_80": mi_final.nlargest(min(80, len(mi_final))).index.tolist(),
        "top_60": mi_final.nlargest(min(60, len(mi_final))).index.tolist(),
        "top_40": mi_final.nlargest(min(40, len(mi_final))).index.tolist(),
        "top_20": mi_final.nlargest(min(20, len(mi_final))).index.tolist(),
    }
    with open(REPORTS_DIR / "feature_selection.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    # ── Áp dụng selection ─────────────────────────────────────────────────
    # Giữ: cột selected + cột không phải feature (giá gốc, target, Date)
    non_feat = [c for c in train.columns if c not in cands]
    out_cols = non_feat + selected

    logger.info("Step 10 done  |  output: %d feature cols + %d non-feature cols",
                len(selected), len(non_feat))

    return (
        train[out_cols].copy(),
        val[out_cols].copy(),
        test[out_cols].copy(),
        config,
    )
