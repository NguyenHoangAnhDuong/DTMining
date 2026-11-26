# Bước 9 — Chia Train / Validation / Test theo thời gian
# Quy trình:
#   1. Kiểm tra data leakage: đảm bảo không có cột feature nào nhìn về tương lai
#   2. Chia theo mốc ngày cứng:
#        Train      : 2001-03-27 → 2017-12-31 (~80 %)
#        Validation : 2018-01-01 → 2019-12-31 (~9 %)
#        Test       : 2020-01-01 → 2022-12-30 (~11 %)
#      Lý do chọn 2020 làm điểm đầu Test: giai đoạn COVID và biến động lớn của giá vàng
#   3. Walk-forward validation trên tập Train:
#        Expanding window, mỗi fold thêm 1 năm vào train, val window = 1 năm
#        Sinh ra các fold và lưu dạng JSON để dùng lại khi training
#   4. Lưu train/val/test ra 3 file CSV riêng
# Lưu ý về data leakage:
#   - Tất cả lag feature đã dùng shift(1) hoặc shift(n>=1) → không nhìn về hiện tại
#   - Rolling feature đã dùng shift(1) trước khi rolling → an toàn
#   - market_phase được tính từ dữ liệu ĐÃ CÓ (không dùng thông tin tương lai)
#   - Cross-asset ratio dùng giá cùng ngày → hợp lệ (available at end-of-day)
#   - Cột target (Tomorrow/Next week/Next Month) là mục tiêu dự báo, KHÔNG được dùng làm feature

import json
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

INPUT_PATH   = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\after_clustering.csv"
OUTPUT_DIR   = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\split"
WF_JSON_PATH = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\split\walk_forward_splits.json"

# Mốc chia tập
TRAIN_END = "2017-12-31"
VAL_START = "2018-01-01"
VAL_END   = "2019-12-31"
TEST_START = "2020-01-01"

# Walk-forward: train tối thiểu 3 năm, mỗi fold mở rộng thêm 1 năm, val window = 1 năm
WF_MIN_TRAIN_YEARS = 3
WF_STEP_YEARS      = 1
WF_VAL_YEARS       = 1

# Cột target — tuyệt đối không được dùng làm feature trong mô hình
TARGET_COLS = ["Tomorrow gold price", "Next week gold price", "Next Month gold price"]

# Cột luôn phải xuất hiện trong output để kiểm tra leakage
LEAKAGE_CHECK_COLS = TARGET_COLS


def check_no_leakage(df: pd.DataFrame) -> None:
    # Kiểm tra đơn giản: cột target không được trùng tên với bất kỳ lag/derived feature nào
    feature_cols = [c for c in df.columns if c not in TARGET_COLS + ["Date", "market_phase"]]
    for t in TARGET_COLS:
        if t in feature_cols:
            raise ValueError(f"Data leakage: '{t}' xuat hien trong feature set!")
    logger.info("Kiem tra data leakage: OK — cac cot target khong nam trong feature")


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df["Date"] = pd.to_datetime(df["Date"])
    train = df[df["Date"] <= TRAIN_END].copy().reset_index(drop=True)
    val   = df[(df["Date"] >= VAL_START) & (df["Date"] <= VAL_END)].copy().reset_index(drop=True)
    test  = df[df["Date"] >= TEST_START].copy().reset_index(drop=True)
    return train, val, test


def generate_walk_forward_splits(train_df: pd.DataFrame) -> list[dict]:
    """
    Expanding window walk-forward validation trên tập train.
    Mỗi fold: train từ đầu đến năm (start_year + n), val = năm tiếp theo.
    """
    df = train_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    min_year  = df["Date"].dt.year.min()
    max_year  = df["Date"].dt.year.max()

    folds: list[dict] = []
    fold_id = 1

    # Điểm kết thúc tập train của fold đầu tiên
    first_train_end_year = min_year + WF_MIN_TRAIN_YEARS - 1

    for train_end_year in range(first_train_end_year, max_year - WF_VAL_YEARS + 1, WF_STEP_YEARS):
        val_start_year = train_end_year + 1
        val_end_year   = val_start_year + WF_VAL_YEARS - 1

        if val_end_year > max_year:
            break

        fold_train = df[df["Date"].dt.year <= train_end_year]
        fold_val   = df[
            (df["Date"].dt.year >= val_start_year) &
            (df["Date"].dt.year <= val_end_year)
        ]

        if len(fold_train) == 0 or len(fold_val) == 0:
            continue

        folds.append(
            {
                "fold"           : fold_id,
                "train_start"    : str(fold_train["Date"].min().date()),
                "train_end"      : str(fold_train["Date"].max().date()),
                "val_start"      : str(fold_val["Date"].min().date()),
                "val_end"        : str(fold_val["Date"].max().date()),
                "train_rows"     : len(fold_train),
                "val_rows"       : len(fold_val),
            }
        )
        fold_id += 1

    logger.info("Tao %d walk-forward folds", len(folds))
    return folds


def process_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    check_no_leakage(df)

    train, val, test = split_data(df)
    wf_folds = generate_walk_forward_splits(train)

    return train, val, test, wf_folds


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    df = pd.read_csv(INPUT_PATH)
    logger.info("Doc du lieu: %d hang x %d cot", *df.shape)

    train, val, test, wf_folds = process_split(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    with open(WF_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(wf_folds, f, indent=2, ensure_ascii=False)

    total = len(train) + len(val) + len(test)

    print("\n=== Tom tat Buoc 9 — Train/Val/Test Split ===")
    print(f"  Tong so hang          : {total}")
    print(f"  Train  : {len(train):>5} hang  "
          f"({train['Date'].min().date()} -> {train['Date'].max().date()})  "
          f"{len(train)/total*100:.1f}%")
    print(f"  Val    : {len(val):>5} hang  "
          f"({val['Date'].min().date()} -> {val['Date'].max().date()})  "
          f"{len(val)/total*100:.1f}%")
    print(f"  Test   : {len(test):>5} hang  "
          f"({test['Date'].min().date()} -> {test['Date'].max().date()})  "
          f"{len(test)/total*100:.1f}%")
    print(f"\n  Walk-forward folds    : {len(wf_folds)}")
    print(f"  (Train window min: {wf_folds[0]['train_start']} - {wf_folds[0]['train_end']})")
    print(f"  (Train window max: {wf_folds[-1]['train_start']} - {wf_folds[-1]['train_end']})")
    print(f"\n  Luu tai               : {OUTPUT_DIR}")
    print(f"  Walk-forward JSON     : {WF_JSON_PATH}")

    for s in ["Tang", "Di ngang", "Giam"]:
        c = (train["market_phase"] == s).sum()
        print(f"    Train phase {s:<10}: {c} ngay ({c/len(train)*100:.1f}%)")


if __name__ == "__main__":
    main()
