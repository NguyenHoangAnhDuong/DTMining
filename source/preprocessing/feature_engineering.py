# Bước 6 — Feature Engineering
# Quy trình:
#   A. Lag features : Gold Price_logret (lag 1,2,3,5,10,20)
#                     + biến ngoại sinh chính (lag 1,2,3,5)
#   B. Rolling stats : mean & std trên Gold Price_logret (cửa sổ 5,10,20,60 ngày)
#                      shift(1) trước khi rolling để không dùng thông tin hiện tại
#   C. Cross-asset   : Gold/Oil ratio, Gold/USD ratio (giá gốc)
#   D. Calendar      : Month, Quarter, DayOfWeek
# Lưu ý:
#   - Feature được tạo trên TOÀN BỘ dataset, không split trước (để rolling đủ dữ liệu)
#   - Hàng warmup có NaN do lag/rolling bị drop ở cuối (tối đa 60 hàng đầu)
#   - Cột target không được dùng để tạo feature (tránh data leakage)

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

INPUT_PATH  = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\after_stationarity.csv"
OUTPUT_PATH = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\after_feature_engineering.csv"

# Cột target — không dùng để tạo feature, không drop hàng dựa vào cột này
TARGET_COLS = {"Tomorrow gold price", "Next week gold price", "Next Month gold price"}

# Lag của Gold Price_logret — chuỗi mục tiêu (đã dừng)
GOLD_LOGRET_LAGS = [1, 2, 3, 5, 10, 20]

# Biến ngoại sinh và số lag tương ứng
EXOG_LAGS: dict[str, list[int]] = {
    "Oil_logret"             : [1, 2, 3, 5],
    "U.S dollar index_logret": [1, 2, 3, 5],
    "S&P 500_logret"         : [1, 2, 3, 5],
    "DJI_logret"             : [1, 2, 3, 5],
    "U.S Treasury_logret"    : [1, 2, 3, 5],
    "Silver_logret"          : [1, 2, 3, 5],
    "Inflation Rate_diff"    : [1, 2, 3, 5],
    "Fedral Funds Rate_diff" : [1, 2, 3, 5],
}

# Cửa sổ rolling (ngày giao dịch)
ROLLING_WINDOWS = [5, 10, 20, 60]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for lag in GOLD_LOGRET_LAGS:
        col = f"Gold Price_logret_lag{lag}"
        df[col] = df["Gold Price_logret"].shift(lag)

    n_gold_lags = len(GOLD_LOGRET_LAGS)
    n_exog_lags = 0
    for base_col, lags in EXOG_LAGS.items():
        if base_col not in df.columns:
            logger.warning("Khong tim thay cot '%s', bo qua lag.", base_col)
            continue
        for lag in lags:
            df[f"{base_col}_lag{lag}"] = df[base_col].shift(lag)
            n_exog_lags += 1

    logger.info("Them %d lag (Gold) + %d lag (exog)", n_gold_lags, n_exog_lags)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    # shift(1) truoc rolling de chi dung du lieu <= t-1, tranh nhin trang lai hien tai
    df = df.copy()
    shifted = df["Gold Price_logret"].shift(1)

    for w in ROLLING_WINDOWS:
        df[f"Gold_logret_roll_mean{w}"] = shifted.rolling(window=w, min_periods=w).mean()
        df[f"Gold_logret_roll_std{w}"]  = shifted.rolling(window=w, min_periods=w).std()

    logger.info("Them rolling mean/std voi cac cua so: %s", ROLLING_WINDOWS)
    return df


def add_cross_asset_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Oil" in df.columns:
        # Dau co the am (thang 4/2020 — gia tuong lai), giu nguyen ratio, model tu hoc
        df["Gold_Oil_ratio"] = df["Gold Price"] / df["Oil"]

    if "U.S dollar index" in df.columns:
        df["Gold_USD_ratio"] = df["Gold Price"] / df["U.S dollar index"]

    logger.info("Them Gold_Oil_ratio, Gold_USD_ratio")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["Date"])
    df["Month"]     = dt.dt.month
    df["Quarter"]   = dt.dt.quarter
    df["DayOfWeek"] = dt.dt.dayofweek
    logger.info("Them Month, Quarter, DayOfWeek")
    return df


def drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Drop hang co NaN o bat ky cot feature nao (tru target va Date)
    feature_cols = [c for c in df.columns if c not in TARGET_COLS | {"Date"}]
    before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    removed = before - len(df)
    logger.info("Drop %d hang warmup (NaN do lag/rolling). Con lai: %d hang", removed, len(df))
    return df


def process_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_cross_asset_ratios(df)
    df = add_calendar_features(df)
    df = drop_warmup_rows(df)

    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    df = pd.read_csv(INPUT_PATH)
    logger.info("Doc du lieu: %d hang x %d cot", *df.shape)

    df_out = process_feature_engineering(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)

    old_cols = set(df.columns)
    new_cols = [c for c in df_out.columns if c not in old_cols]

    print("\n=== Tom tat Buoc 6 — Feature Engineering ===")
    print(f"  So hang truoc      : {len(df)}")
    print(f"  So hang sau        : {len(df_out)}")
    print(f"  Hang bi drop       : {len(df) - len(df_out)} (warmup NaN)")
    print(f"  So cot cu          : {len(old_cols)}")
    print(f"  So cot them moi    : {len(new_cols)}")
    print(f"  Tong so cot        : {df_out.shape[1]}")
    print(f"  Missing con lai    : {df_out.isnull().sum().sum()}")
    print(f"  Pham vi ngay       : {df_out['Date'].min()} -> {df_out['Date'].max()}")
    print(f"  Luu tai            : {OUTPUT_PATH}")
    print(f"\nCac cot them moi ({len(new_cols)} cot):")
    for c in new_cols:
        print(f"  {c}")


if __name__ == "__main__":
    main()
