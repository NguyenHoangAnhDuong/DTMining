"""
Khoảng dự đoán (prediction interval) theo phương pháp split-conformal trên log-return.

Tập hiệu chỉnh: sai số out-of-fold walk-forward trên Train (split = "cv", 2004–2017),
có đủ mọi giai đoạn thị trường (gồm cả Volatile 2008, 2013).

  - Nhánh base   : một độ rộng chung q_α = quantile(|e|, α) cho mọi ngày.
  - Nhánh kmeans : độ rộng theo giai đoạn q_α(regime) — model K-Means gán giai đoạn trước,
                   giai đoạn biến động mạnh → khoảng rộng hơn, giai đoạn êm → hẹp hơn.

Khoảng giá: [P_t · exp(ŷ − q), P_t · exp(ŷ + q)]

Đánh giá: Coverage (% giá thực nằm trong khoảng, càng sát mức danh nghĩa càng tốt),
độ rộng trung bình (USD, hẹp hơn là tốt hơn nếu coverage đạt),
Winkler / interval score (gộp cả hai, càng nhỏ càng tốt).
"""
import numpy as np
import pandas as pd

LEVELS = [0.80, 0.95]
MIN_REGIME_SAMPLES = 30


def _q(abs_err: np.ndarray, level: float) -> float:
    n = len(abs_err)
    k = min(1.0, np.ceil((n + 1) * level) / n)   # hiệu chỉnh mẫu hữu hạn của conformal
    return float(np.quantile(abs_err, k))


def fit_widths(cal: pd.DataFrame, conditional: bool) -> dict:
    """cal: dự đoán out-of-fold (cột lr_true, lr_pred, regime_label). Trả về {level: {regime|None: q}}."""
    err = (cal["lr_true"] - cal["lr_pred"]).abs()
    widths = {}
    for lv in LEVELS:
        widths[lv] = {None: _q(err.to_numpy(), lv)}
        if conditional:
            for r, e in err.groupby(cal["regime_label"]):
                if len(e) >= MIN_REGIME_SAMPLES:
                    widths[lv][r] = _q(e.to_numpy(), lv)
    return widths


def apply_widths(df: pd.DataFrame, widths: dict, level: float) -> tuple[np.ndarray, np.ndarray]:
    w = widths[level]
    q = df["regime_label"].map(lambda r: w.get(r, w[None])).to_numpy(dtype=float)
    lr = df["lr_pred"].to_numpy()
    p = df["price_t"].to_numpy()
    return p * np.exp(lr - q), p * np.exp(lr + q)


def winkler(y, lo, hi, level: float) -> np.ndarray:
    a = 1 - level
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return (hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)


def interval_scores(y, lo, hi, level: float) -> dict:
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return {
        "coverage_%": float(np.mean((y >= lo) & (y <= hi)) * 100),
        "width_usd":  float(np.mean(hi - lo)),
        "winkler":    float(np.mean(winkler(y, lo, hi, level))),
    }
