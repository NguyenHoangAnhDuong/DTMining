"""
Step 06 — Multi-Horizon Target Engineering
  Tính lại target từ Gold Price sau khi đã làm sạch hoàn toàn.
  Tại sao tính lại thay vì dùng cột "Tomorrow gold price" gốc?
  → Dữ liệu gốc tính target trước khi xóa weekend và clean, có thể không khớp sau pipeline.

  Với mỗi horizon h ∈ {1, 7, 30} (đơn vị: ngày giao dịch):
    target_price_h{h}   = Gold Price t+h                (giá tuyệt đối)
    target_logret_h{h}  = ln(Gold Price t+h / Gold Price t)  (log-return h ngày)
    target_date_h{h}    = Date t+h                      (ngày tương lai, để purge split)

  Các cột target gốc từ raw data ("Tomorrow gold price", ...) được GIỮ LẠI để tương thích
  với XGBoost model cũ. Model mới nên dùng target_price_h{h}.
"""
import logging

import numpy as np
import pandas as pd

from .config import HORIZONS

logger = logging.getLogger(__name__)


def run(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 06 — Multi-Horizon Targets  |  h = %s", HORIZONS)
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    price = df["Gold Price"]

    for h in HORIZONS:
        future_price = price.shift(-h)
        df[f"target_price_h{h}"]  = future_price
        df[f"target_logret_h{h}"] = np.log(future_price / price)
        df[f"target_date_h{h}"]   = df["Date"].shift(-h)

    h_max = max(HORIZONS)
    n_nan_tail = df[f"target_price_h{h_max}"].isna().sum()
    logger.info(
        "Step 06 done  |  %d tail rows have NaN targets (h=%d) — kept, models will handle",
        n_nan_tail, h_max,
    )
    return df
