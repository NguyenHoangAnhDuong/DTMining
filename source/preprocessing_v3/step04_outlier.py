"""
Step 04 — Outlier Detection  (KHÔNG capping, chỉ báo cáo)
  Lý do KHÔNG capping: biến động cực đoan trong tài chính là sự kiện thật
  (giá dầu âm tháng 4/2020, khủng hoảng tài chính 2008, v.v.).
  Capping sẽ làm mô hình không học được phản ứng thị trường trong giai đoạn khủng hoảng.

  Xuất:
    - outlier_summary.csv  : ngưỡng IQR + số lượng ngoại lai theo cột
    - outlier_details.csv  : từng điểm ngoại lai kèm ngày, giá trị, mức độ
    - outlier_boxplots.png : boxplot toàn bộ cột MARKET + MACRO
"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES_DIR, MACRO, MARKET, REPORTS_DIR

logger = logging.getLogger(__name__)


def run(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 04 — Outlier Detection (detect only, NO modification)")
    df = df.copy()

    cols    = [c for c in MARKET + MACRO if c in df.columns]
    rows    = []
    details = []

    for col in cols:
        series = df[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr     = q3 - q1
        lo,  hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        xlo, xhi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        mask = (df[col] < lo) | (df[col] > hi)

        for idx in df.index[mask]:
            val = df.at[idx, col]
            details.append({
                "Date"    : df.at[idx, "Date"],
                "feature" : col,
                "value"   : val,
                "severity": "extreme" if (val < xlo or val > xhi) else "mild",
            })

        rows.append({
            "feature"    : col,
            "q1"         : round(q1, 4),
            "q3"         : round(q3, 4),
            "lower_bound": round(lo, 4),
            "upper_bound": round(hi, 4),
            "n_outliers" : int(mask.sum()),
        })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "outlier_summary.csv", index=False)
    pd.DataFrame(details).to_csv(REPORTS_DIR / "outlier_details.csv", index=False)

    ncols = 4
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.5))
    for ax, col in zip(axes.flat, cols):
        ax.boxplot(df[col].dropna(), patch_artist=True,
                   boxprops=dict(facecolor="#cce5ff"),
                   medianprops=dict(color="#003366"))
        ax.set_title(col, fontsize=8)
    for ax in list(axes.flat)[len(cols):]:
        ax.set_visible(False)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "outlier_boxplots.png", dpi=120, bbox_inches="tight")
    plt.close()

    n_total = sum(r["n_outliers"] for r in rows)
    logger.info("Step 04 done  |  %d outliers detected across %d cols (data unchanged)",
                n_total, len(cols))
    return df
