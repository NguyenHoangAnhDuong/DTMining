"""Dự đoán cho web app — không phụ thuộc Streamlit (có thể dùng lại cho API khác)."""
import importlib
from functools import lru_cache

import pandas as pd

from source.services.constants import MODEL_MODULES

# predict_latest cần ≥ 200 phiên (MA200); lấy dư để rolling ổn định
HISTORY_ROWS = 260


@lru_cache(maxsize=None)
def _predictor(model: str):
    if model not in MODEL_MODULES:
        raise ValueError(f"Mô hình chưa hỗ trợ dự đoán: {model}")
    return importlib.import_module(MODEL_MODULES[model]).predict_latest


def min_anchor_index() -> int:
    return HISTORY_ROWS - 1


def forecast_at(data: pd.DataFrame, model: str, branch: str, horizon: int, anchor_idx: int) -> dict:
    """
    Dự đoán tại phiên `anchor_idx` của base_dataset (chỉ dùng dữ liệu đến phiên đó).
    Nếu dữ liệu có giá thực tế sau h phiên thì trả kèm để đối chiếu.
    """
    if anchor_idx < min_anchor_index():
        raise ValueError("Không đủ lịch sử để tạo feature (cần ≥ 260 phiên).")

    history = data.iloc[anchor_idx - HISTORY_ROWS + 1: anchor_idx + 1]
    result = _predictor(model)(history, branch, horizon)

    row = data.iloc[anchor_idx]
    actual = row.get(f"target_price_h{horizon}")
    target_date = row.get(f"target_date_h{horizon}")
    has_actual = pd.notna(actual)

    if not has_actual or pd.isna(target_date):
        target_date = pd.Timestamp(result["date"]) + pd.offsets.BDay(horizon)

    out = dict(result)
    out.update(
        model=model,
        date=pd.Timestamp(result["date"]),
        target_date=pd.Timestamp(target_date),
        actual=float(actual) if has_actual else None,
        intervals={lv: (float(lo), float(hi)) for lv, (lo, hi) in result["intervals"].items()},
    )
    if has_actual:
        out["error"] = out["pred_price"] - out["actual"]
        out["inside"] = {lv: lo <= out["actual"] <= hi for lv, (lo, hi) in out["intervals"].items()}
    return out


def result_to_frame(results: list[dict]) -> pd.DataFrame:
    """Chuyển danh sách kết quả dự đoán thành bảng (để hiển thị / tải về)."""
    rows = []
    for r in results:
        row = {
            "model": r["model"], "branch": r["branch"], "horizon": r["horizon"],
            "date": r["date"].date(), "target_date": r["target_date"].date(),
            "regime": r["regime"], "price_t": r["price_t"], "pred_price": r["pred_price"],
            "pred_return_%": r["pred_logret"] * 100,
        }
        for lv, (lo, hi) in r["intervals"].items():
            pct = int(float(lv) * 100)
            row[f"lo_{pct}"], row[f"hi_{pct}"] = lo, hi
        row["actual"] = r.get("actual")
        row["error"] = r.get("error")
        rows.append(row)
    return pd.DataFrame(rows)
