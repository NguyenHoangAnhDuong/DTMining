"""Chỉ số đánh giá dùng chung cho mọi model (tính trên GIÁ sau khi quy đổi từ log-return)."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

METRIC_COLS = ["MAE", "RMSE", "MAPE", "R2", "MASE", "DA"]


def forecast_metrics(price_t, true_price, pred_price, true_lr, pred_lr) -> dict:
    """
    MASE: chia cho MAE của naive h bước (giá t+h = giá t) trên cùng mẫu → MASE < 1 là tốt hơn naive.
    DA  : % dự đoán đúng chiều tăng/giảm của giá t+h so với giá t.
    """
    price_t, true_price, pred_price = map(np.asarray, (price_t, true_price, pred_price))
    mae   = float(mean_absolute_error(true_price, pred_price))
    naive = float(np.mean(np.abs(true_price - price_t)))
    return {
        "MAE":  mae,
        "RMSE": float(np.sqrt(mean_squared_error(true_price, pred_price))),
        "MAPE": float(np.mean(np.abs((true_price - pred_price) / true_price)) * 100),
        "R2":   float(r2_score(true_price, pred_price)),
        "MASE": mae / naive if naive > 0 else np.nan,
        "DA":   float(np.mean(np.sign(np.asarray(pred_lr)) == np.sign(np.asarray(true_lr))) * 100),
        "n":    int(len(true_price)),
    }


def diebold_mariano(err_a, err_b, horizon: int) -> tuple[float, float]:
    """
    Kiểm định Diebold-Mariano (loss = |e|, phương sai HAC với lag h-1, hiệu chỉnh Harvey 1997).
    stat < 0 → model A có sai số nhỏ hơn B. Trả về (stat, p-value hai phía).
    """
    return dm_on_losses(np.abs(np.asarray(err_a)), np.abs(np.asarray(err_b)), horizon)


def dm_on_losses(loss_a, loss_b, horizon: int) -> tuple[float, float]:
    """Diebold-Mariano trên chuỗi loss bất kỳ (vd. Winkler score). stat < 0 → A tốt hơn."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    d_mean = d.mean()
    gamma = [np.sum((d[k:] - d_mean) * (d[:n - k] - d_mean)) / n for k in range(horizon)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = d_mean / np.sqrt(var)
    adj = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = dm * adj
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def metrics_table(preds: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """metrics.csv chuẩn cho một model (mọi nhánh × h × split) + dòng naive trên cùng các dòng."""
    rows = []
    for (branch, h, sp), g in preds.groupby(["branch", "horizon", "split"]):
        rows.append({"model": model_name, "branch": branch, "horizon": h, "split": sp,
                     **forecast_metrics(g["price_t"], g["y_true"], g["y_pred"], g["lr_true"], g["lr_pred"])})
    for (h, sp), g in preds[preds["branch"] == "base"].groupby(["horizon", "split"]):
        m = forecast_metrics(g["price_t"], g["y_true"], g["price_t"], g["lr_true"], np.zeros(len(g)))
        m["DA"] = np.nan   # naive không dự đoán chiều
        rows.append({"model": "naive", "branch": "naive", "horizon": h, "split": sp, **m})
    return pd.DataFrame(rows).sort_values(["split", "horizon", "branch"]).reset_index(drop=True)


def metrics_by_group(pred: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, g in pred.groupby(group_col):
        rows.append({group_col: key, **forecast_metrics(g["price_t"], g["y_true"], g["y_pred"],
                                                         g["lr_true"], g["lr_pred"])})
    return pd.DataFrame(rows)
