# Bước 4 — làm sạch cấu trúc chuỗi thời gian
# Việc cần làm:
#   1. Chuyển cột Date từ string sang datetime64
#   2. Xóa 2288 hàng cuối tuần (Thứ 7, CN) — giá vàng cuối tuần chỉ là copy của Thứ Sáu,
#      giữ lại sẽ làm sai ACF/PACF và các chỉ số rolling ở bước sau
#   3. Sắp xếp theo ngày và reset index
#   4. Ghi log các khoảng trống ngày lễ (>= 4 ngày lịch) — giữ nguyên, không fill,
#      vì thị trường đóng cửa thực sự thì không có dữ liệu để fill
# Kết quả: 8021 hàng -> 5733 hàng (chỉ giữ ngày làm việc Thứ 2 đến Thứ 6)

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

INPUT_PATH  = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\after_outlier.csv"
OUTPUT_PATH = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\after_time_series.csv"


def _parse_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    logger.info("Chuyen Date sang datetime64. Pham vi: %s -> %s",
                df["Date"].min().date(), df["Date"].max().date())
    return df


def _remove_weekends(df: pd.DataFrame) -> pd.DataFrame:
    # Giá vàng Thứ 7/CN = giá Thứ Sáu, nếu giữ lại sẽ làm rolling stats bị kéo về giá cũ,
    # ACF/PACF sẽ có spike giả tạo ở lag 2 mỗi tuần
    before = len(df)
    df = df[df["Date"].dt.dayofweek < 5].copy()   # 0=Mon ... 4=Fri
    logger.info("Da xoa %d hang cuoi tuan. Con lai: %d hang", before - len(df), len(df))
    return df


def _sort_reset(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("Date").reset_index(drop=True)


def _log_holiday_gaps(df: pd.DataFrame, threshold_days: int = 4) -> None:
    # In ra các khoảng trống ngày lễ để biết thị trường đóng cửa hôm nào — không xử lý gì thêm
    diffs = df["Date"].diff().dropna()
    gaps  = diffs[diffs.dt.days >= threshold_days]
    if gaps.empty:
        logger.info("Khong co khoang trong ngay le >= %d ngay.", threshold_days)
        return
    logger.info("Cac khoang trong ngay le >= %d ngay (%d truong hop):", threshold_days, len(gaps))
    for idx in gaps.index:
        b = df.loc[idx - 1, "Date"]
        a = df.loc[idx,     "Date"]
        logger.info("  %s (%s) -> %s (%s)  [%d ngay]",
                    b.date(), b.day_name(), a.date(), a.day_name(), diffs.loc[idx].days)


def _check_integrity(df: pd.DataFrame) -> None:
    # Kiểm tra lần cuối xem còn ngày trùng hoặc NaT không
    n_dup = df["Date"].duplicated().sum()
    if n_dup:
        logger.warning("Van con %d ngay trung lap!", n_dup)
    else:
        logger.info("Khong co ngay trung lap.")

    if df["Date"].isna().any():
        logger.warning("Co gia tri NaT trong cot Date!")
    else:
        logger.info("Cot Date khong co NaT.")

    logger.info("Kich thuoc cuoi: %d hang x %d cot", *df.shape)


def process_time_series(df: pd.DataFrame) -> pd.DataFrame:
    df = _parse_date(df)
    df = _remove_weekends(df)
    df = _sort_reset(df)
    _log_holiday_gaps(df)
    _check_integrity(df)
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    df = pd.read_csv(INPUT_PATH)
    logger.info("Doc du lieu: %d hang x %d cot", *df.shape)

    df_out = process_time_series(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    logger.info("Luu tai: %s", OUTPUT_PATH)

    print("\n=== Tom tat Buoc 4 ===")
    print(f"  Truoc xu ly : {len(df)} hang")
    print(f"  Sau xu ly   : {len(df_out)} hang")
    print(f"  Da xoa      : {len(df) - len(df_out)} hang (cuoi tuan)")
    print(f"  Pham vi ngay: {df_out['Date'].min()} -> {df_out['Date'].max()}")
    print(f"  So cot      : {df_out.shape[1]}")
    print(f"  Missing     : {df_out.isnull().sum().sum()}")
    print(f"  Luu tai     : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
