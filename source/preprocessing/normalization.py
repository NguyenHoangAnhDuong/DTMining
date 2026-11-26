# Bước 10 — Chuẩn hóa dữ liệu (StandardScaler fit chỉ trên Train)
# Quy trình:
#   1. Xác định cột cần scale: 54 ML feature đã qua Feature Selection
#      (loại trừ: Date, market_phase, target, giá gốc không dừng)
#   2. Fit StandardScaler CHỈ trên tập train — không nhìn vào val/test
#   3. Transform train, val, test bằng scaler đã fit
#   4. Lưu scaler ra file pkl để dùng lại khi inference
#   5. Lưu 3 file CSV đã scale (thay thế giá trị feature, giữ nguyên các cột khác)
# Lưu ý:
#   - Giá gốc (Gold Price, Oil, v.v.) KHÔNG scale ở bước này —
#     SARIMAX dùng trực tiếp, LSTM sẽ scale riêng trong pipeline model
#   - market_phase là chuỗi phân loại, không scale
#   - Cột target không scale (đây là giá trị dự báo, model sẽ dự báo ở thang gốc)
#   - Sau khi scale: mean ≈ 0, std ≈ 1 trên tập train; val/test có thể lệch nhẹ

import logging
import os
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

TRAIN_PATH  = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\split\train.csv"
VAL_PATH    = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\split\val.csv"
TEST_PATH   = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\split\test.csv"
REPORT_PATH = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\reports\feature_selection_report.csv"

OUTPUT_DIR  = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\split"
SCALER_PATH = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\models\scaler.pkl"

# Cột không được scale (giữ nguyên)
NO_SCALE = {
    "Date", "market_phase",
    "Tomorrow gold price", "Next week gold price", "Next Month gold price",
    # Giá gốc không dừng
    "Gold Price", "Oil", "S&P 500", "DJI", "U.S Treasury",
    "U.S dollar index", "Inflation Rate", "CPI Rate", "Silver",
    "Copper", "Iron Price", "Platinum", "Palladium", "Fedral Funds Rate",
}


def _get_feature_cols(df: pd.DataFrame, report_path: str) -> list[str]:
    # Lấy danh sách feature từ báo cáo Feature Selection (Status == Kept)
    # Lọc thêm để chỉ giữ cột thực sự tồn tại trong df và không thuộc NO_SCALE
    report = pd.read_csv(report_path)
    kept   = report.loc[report["Status"] == "Kept", "Feature"].tolist()
    cols   = [c for c in kept if c in df.columns and c not in NO_SCALE]
    return cols


def process_normalization(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()

    # Fit CHỈ trên train
    scaler.fit(train[feature_cols])

    train_out = train.copy()
    val_out   = val.copy()
    test_out  = test.copy()

    train_out[feature_cols] = scaler.transform(train[feature_cols])
    val_out[feature_cols]   = scaler.transform(val[feature_cols])
    test_out[feature_cols]  = scaler.transform(test[feature_cols])

    return train_out, val_out, test_out, scaler


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    train = pd.read_csv(TRAIN_PATH)
    val   = pd.read_csv(VAL_PATH)
    test  = pd.read_csv(TEST_PATH)
    logger.info("Doc train=%d  val=%d  test=%d hang", len(train), len(val), len(test))

    feature_cols = _get_feature_cols(train, REPORT_PATH)
    logger.info("So cot se scale: %d", len(feature_cols))

    train_sc, val_sc, test_sc, scaler = process_normalization(
        train, val, test, feature_cols
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_sc.to_csv(os.path.join(OUTPUT_DIR, "train_scaled.csv"), index=False)
    val_sc.to_csv(os.path.join(OUTPUT_DIR,   "val_scaled.csv"),   index=False)
    test_sc.to_csv(os.path.join(OUTPUT_DIR,  "test_scaled.csv"),  index=False)

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # Kiểm tra nhanh: mean và std của feature trên train sau khi scale
    means = train_sc[feature_cols].mean()
    stds  = train_sc[feature_cols].std()
    max_abs_mean = means.abs().max()
    min_std      = stds.min()
    max_std      = stds.max()

    print("\n=== Tom tat Buoc 10 — Normalization ===")
    print(f"  So feature duoc scale  : {len(feature_cols)}")
    print(f"  Kiem tra train sau scale:")
    print(f"    max |mean|           : {max_abs_mean:.2e}  (ky vong ~ 0)")
    print(f"    std  min / max       : {min_std:.4f} / {max_std:.4f}  (ky vong ~ 1)")
    print(f"  Scaler luu tai         : {SCALER_PATH}")
    print(f"  train_scaled.csv       : {os.path.join(OUTPUT_DIR, 'train_scaled.csv')}")
    print(f"  val_scaled.csv         : {os.path.join(OUTPUT_DIR, 'val_scaled.csv')}")
    print(f"  test_scaled.csv        : {os.path.join(OUTPUT_DIR, 'test_scaled.csv')}")


if __name__ == "__main__":
    main()
