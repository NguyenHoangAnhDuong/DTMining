"""
Step 05 — Stationarity Tests & Transforms
  1. Kiểm định ADF + KPSS trên chuỗi gốc (level)
  2. Thêm cột transform:
       - {col}_logret = ln(P_t / P_{t-1})  cho cột giá luôn dương  (LOGRET_COLS)
       - {col}_diff   = P_t - P_{t-1}       cho cột có thể âm/bằng 0 (DIFF_COLS)
       Đặc biệt: Oil dùng _diff (tháng 4/2020 Oil âm, không lấy log được)
  3. Kiểm định lại sau chuyển đổi để xác nhận tính dừng
  4. Lưu báo cáo CSV, vẽ ACF/PACF (Gold log-return) và phân rã STL (Gold Price)

  Kết luận kiểm định — thận trọng hơn v2:
    "stationary"    : ADF bác bỏ H0 AND KPSS không bác bỏ H0
    "nonstationary" : ADF không bác bỏ H0 AND KPSS bác bỏ H0
    "inconclusive"  : hai kiểm định mâu thuẫn (không tự suy luận)
"""
import logging
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from .config import DIFF_COLS, FIGURES_DIR, LOGRET_COLS, REPORTS_DIR

logger = logging.getLogger(__name__)

STL_PERIOD = 252   # ~252 ngày giao dịch mỗi năm


# ── Kiểm định ─────────────────────────────────────────────────────────────

def _test_one(series: pd.Series, name: str, transform: str) -> dict:
    s = series.dropna()
    base = {"series": name, "transform": transform, "n": len(s)}
    if len(s) < 30 or s.nunique() < 2:
        return {**base, "conclusion": "insufficient"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf = adfuller(s, autolag="AIC")
            kss = kpss(s, regression="c", nlags="auto")
        a_reject = adf[1] < 0.05
        k_reject = kss[1] < 0.05
        if a_reject and not k_reject:
            conclusion = "stationary"
        elif not a_reject and k_reject:
            conclusion = "nonstationary"
        else:
            conclusion = "inconclusive"
        return {
            **base,
            "adf_stat": round(adf[0], 4), "adf_p": round(adf[1], 4),
            "kpss_stat": round(kss[0], 4), "kpss_p": round(kss[1], 4),
            "conclusion": conclusion,
        }
    except Exception as exc:
        return {**base, "conclusion": "failed", "error": str(exc)}


# ── Transforms ────────────────────────────────────────────────────────────

def _add_logret(df: pd.DataFrame) -> pd.DataFrame:
    for col in LOGRET_COLS:
        if col not in df.columns:
            continue
        # where(>0) bảo vệ nếu còn giá trị <= 0 sót lại (không nên có, nhưng an toàn)
        pos = df[col].where(df[col] > 0)
        df[col + "_logret"] = np.log(pos) - np.log(pos.shift(1))
        logger.info("Added '%s_logret'", col)
    return df


def _add_diffs(df: pd.DataFrame) -> pd.DataFrame:
    for col in DIFF_COLS:
        if col not in df.columns:
            continue
        df[col + "_diff"] = df[col].diff()
        logger.info("Added '%s_diff'", col)
    return df


# ── Visualisations ────────────────────────────────────────────────────────

def _plot_acf_pacf(series: pd.Series, title: str, path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plot_acf(series.dropna(),  lags=50, ax=ax1, alpha=0.05, zero=False)
    ax1.set_title("ACF — xác định bậc MA (q)", fontsize=10)
    plot_pacf(series.dropna(), lags=50, ax=ax2, alpha=0.05, method="ywm", zero=False)
    ax2.set_title("PACF — xác định bậc AR (p)", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved ACF/PACF → %s", path)


def _plot_stl(series: pd.Series, period: int, title: str, path) -> None:
    stl = STL(series.dropna(), period=period, robust=True)
    res = stl.fit()
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    for ax, data, label, color in zip(
        axes,
        [series.values, res.trend, res.seasonal, res.resid],
        ["Original", "Trend", "Seasonal", "Residual"],
        ["#2176ae", "#e63946", "#2a9d8f", "#6c757d"],
    ):
        ax.plot(series.index, data, linewidth=0.7, color=color)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved STL → %s", path)


# ── Main ──────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 05 — Stationarity Tests & Transforms")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    test_cols = [c for c in LOGRET_COLS + DIFF_COLS if c in df.columns]

    rows_before = [_test_one(df[c], c, "level") for c in test_cols]

    df = _add_logret(df)
    df = _add_diffs(df)

    transformed = (
        [c + "_logret" for c in LOGRET_COLS if c in df.columns] +
        [c + "_diff"   for c in DIFF_COLS   if c in df.columns]
    )
    rows_after = [_test_one(df[c], c, "transformed") for c in transformed if c in df.columns]

    report = pd.DataFrame(rows_before + rows_after)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORTS_DIR / "stationarity_report.csv", index=False)

    stationary = sum(r["conclusion"] == "stationary" for r in rows_after)
    logger.info("After transforms: %d / %d series stationary", stationary, len(rows_after))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if "Gold Price_logret" in df.columns:
        lr = df["Gold Price_logret"].dropna()
        lr.index = df.loc[lr.index, "Date"]
        _plot_acf_pacf(lr,
                       title="ACF / PACF — Gold Price Log-Return",
                       path=FIGURES_DIR / "acf_pacf_gold_logret.png")

    if "Gold Price" in df.columns:
        gp = df["Gold Price"].copy()
        gp.index = pd.to_datetime(df["Date"])
        _plot_stl(gp, period=STL_PERIOD,
                  title="STL Decomposition — Gold Price (original)",
                  path=FIGURES_DIR / "stl_gold_price.png")

    new_cols = [c for c in transformed if c in df.columns]
    logger.info("Step 05 done  |  added %d transform cols", len(new_cols))
    return df
