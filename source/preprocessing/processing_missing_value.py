# Bước 3a — xử lý missing values trong dataset
# Các vấn đề đã phát hiện qua EDA:
#   - 2 ngày không tồn tại: '6/31/2006', '6/31/2005' (tháng 6 chỉ có 30 ngày) -> xóa luôn
#   - 6 ngày bị trùng: 4 hàng giống hệt + 2 hàng xung đột giá -> giữ hàng cuối cùng
#   - 6 ô có ký tự '-' ẩn thay cho NaN trong cột 'U.S dollar index' -> chuyển thành NaN rồi nội suy
#   - 1 ô '-' ẩn trong cột 'Palladium' trùng ngày không hợp lệ -> tự xóa cùng hàng đó
#   - Giá dầu âm tháng 4/2020: đây là sự kiện thị trường tương lai thực sự -> hợp lệ, giữ nguyên
#   - Lạm phát âm hoặc bằng 0: giai đoạn giảm phát thực sự -> hợp lệ, giữ nguyên

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INPUT_PATH  = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\data_root\dataset_root.csv"
OUTPUT_PATH = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\after_missing.csv"

# Các cột này là cờ sự kiện nhị phân (0/1) — dùng ffill để giữ trạng thái trước đó,
# thiếu ở đầu file thì fill 0
CATEGORICAL_FLAG_COLS: list[str] = [
    "Economic Data", "logistic", "Politics", "oil", "OPEC",
    "Stock (US)", "Stock (Global)",
    "currency (Dollar)", "currency (Russian ruble)",
    "currency (Sterling pound)", "currency (Euro)", "currency (Yen)",
    "Inflation", "Job report", "unemployment rate",
    "CPI", "Interest Rate", "Bonds",
]

# Các cột này là giá hoặc chỉ số liên tục — dùng nội suy theo thời gian
# thay vì copy giá trị cũ như ffill
CONTINUOUS_PRICE_COLS: list[str] = [
    "Gold Price", "Tomorrow gold price", "Next week gold price", "Next Month gold price",
    "Oil", "S&P 500", "DJI", "U.S Treasury", "U.S dollar index",
    "Inflation Rate", "CPI Rate", "Silver", "Copper", "Iron Price",
    "Platinum", "Palladium", "Fedral Funds Rate",
]

MAX_INTERP_GAP: int = 3   # nội suy tối đa 3 ngày liên tiếp, quá 3 ngày thì dùng ffill/bfill
_HIDDEN_NA: str = "-"     # ký tự '-' là NaN ẩn trong data gốc, không phải số âm


def _fix_invalid_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Chuyển cột Date sang kiểu datetime, cái nào không parse được thì thành NaT rồi xóa
    raw_dates = df["Date"].copy()
    df["Date"] = pd.to_datetime(
        df["Date"], format="mixed", dayfirst=False, errors="coerce"
    )
    nat_mask = df["Date"].isna()
    if nat_mask.any():
        bad = raw_dates[nat_mask].tolist()
        logger.warning("Xoa %d hang co ngay khong hop le: %s", len(bad), bad)
        df = df.loc[~nat_mask].copy()
    return df


def _dedup_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Nếu cùng ngày xuất hiện nhiều lần thì giữ lại hàng cuối cùng
    n_before = len(df)
    df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    removed = n_before - len(df)
    if removed:
        logger.info("Xoa %d hang trung lap — giu hang cuoi cung", removed)
    return df


def _resolve_hidden_na(df: pd.DataFrame) -> pd.DataFrame:
    # Duyệt các cột kiểu string, chỗ nào là '-' thì đổi thành NaN, còn lại ép về số
    str_cols = df.select_dtypes(include="object").columns.difference(["Date"])
    for col in str_cols:
        mask = df[col].astype(str).str.strip() == _HIDDEN_NA
        if mask.any():
            logger.info("Cot '%s': thay %d ky tu '-' bang NaN", col, mask.sum())
            df[col] = df[col].where(~mask, other=np.nan)
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() > 0:
            df[col] = coerced
    return df


def _impute_categorical_flags(df: pd.DataFrame) -> pd.DataFrame:
    # Với cột cờ sự kiện: ffill lan giá trị trước đó xuống,
    # chỗ nào đầu file vẫn NaN thì fill 0
    present = [c for c in CATEGORICAL_FLAG_COLS if c in df.columns]
    for col in present:
        n_missing = df[col].isna().sum()
        if n_missing:
            logger.info("Co '%s': dien %d NaN bang ffill roi fill 0", col, n_missing)
            df[col] = df[col].ffill().fillna(0).astype(int)
    return df


def _impute_continuous_prices(df: pd.DataFrame) -> pd.DataFrame:
    # Nội suy theo thời gian cho cột giá: giống vẽ đường thẳng giữa 2 điểm biết trước
    # Nếu khoảng trống quá 3 ngày thì dùng ffill/bfill thay thế
    present = [c for c in CONTINUOUS_PRICE_COLS if c in df.columns]
    df = df.set_index("Date").sort_index()
    for col in present:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        logger.info("Lien tuc '%s': noi suy %d NaN (time -> ffill -> bfill)", col, n_missing)
        df[col] = (
            df[col]
            .interpolate(method="time", limit=MAX_INTERP_GAP, limit_direction="forward")
            .ffill()
            .bfill()
        )
    return df.reset_index()


def process_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Chạy tuần tự 5 bước: sửa ngày -> bỏ trùng -> đổi '-' thành NaN -> fill cờ -> nội suy giá
    df = df.copy()
    df = _fix_invalid_dates(df)
    df = _dedup_dates(df)
    df = _resolve_hidden_na(df)
    df = _impute_categorical_flags(df)
    df = _impute_continuous_prices(df)

    remaining = df.isnull().sum().sum()
    if remaining:
        logger.warning("%d NaN con sot lai sau khi xu ly", remaining)
    else:
        logger.info("Xu ly missing values hoan tat.")

    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    df = pd.read_csv(INPUT_PATH)
    print(f"Doc du lieu: {df.shape[0]} hang x {df.shape[1]} cot")

    missing_before = df.isnull().sum().sum()
    print(f"Missing truoc xu ly : {missing_before}")

    df_out = process_missing_values(df)

    missing_after = df_out.isnull().sum().sum()
    print(f"Missing sau xu ly  : {missing_after}")
    print(f"Kich thuoc dau ra  : {df_out.shape}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"Luu tai: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
