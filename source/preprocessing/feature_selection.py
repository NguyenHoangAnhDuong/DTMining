# Bước 7 — Feature Selection
# Quy trình:
#   1. Xác định cột feature ứng viên (loại trừ target, Date, giá gốc không dừng)
#   2. Lọc tương quan cao: với mỗi cặp |corr| > 0.95, giữ cột có MI cao hơn, loại còn lại
#   3. Tính Mutual Information (MI) của từng feature với target = log-return 1 ngày tới
#   4. Lọc theo MI: giữ feature có MI > 0 (loại feature không mang thông tin)
#   5. Xuất báo cáo CSV: MI score, trạng thái kept/removed, lý do loại
#   6. Lưu dataset cuối: giữ feature đã chọn + giá gốc + target + Date
# Lưu ý:
#   - Target để tính MI = log(Tomorrow gold price / Gold Price) = log-return 1 ngày tới
#   - Giá gốc không dừng (Oil, S&P 500, v.v.) KHÔNG tham gia lọc nhưng được giữ lại trong
#     dataset output vì SARIMAX cần dùng trực tiếp
#   - Bước này chạy trên TOÀN BỘ dataset — khi đến bước 9 sẽ fit lại feature selector
#     chỉ trên tập train để tránh data leakage thực sự

import logging
import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)

INPUT_PATH   = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\after_feature_engineering.csv"
OUTPUT_CSV   = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\after_feature_selection.csv"
REPORT_CSV   = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\reports\feature_selection_report.csv"

# Cột không tham gia lọc feature (giữ lại trong output nhưng không đánh giá)
DATE_COL    = "Date"
TARGET_COLS = {"Tomorrow gold price", "Next week gold price", "Next Month gold price"}

# Giá gốc không dừng: giữ lại cho SARIMAX nhưng loại khỏi pool feature cho ML
RAW_PRICE_COLS = {
    "Gold Price", "Oil", "S&P 500", "DJI", "U.S Treasury",
    "U.S dollar index", "Inflation Rate", "CPI Rate", "Silver",
    "Copper", "Iron Price", "Platinum", "Palladium", "Fedral Funds Rate",
}

CORR_THRESHOLD = 0.95   # Ngưỡng tương quan tuyến tính (Pearson)


def _build_target(df: pd.DataFrame) -> pd.Series:
    # log-return 1 ngày tới: log(P_{t+1} / P_t)
    return np.log(df["Tomorrow gold price"] / df["Gold Price"])


def _get_feature_candidates(df: pd.DataFrame) -> list[str]:
    exclude = TARGET_COLS | RAW_PRICE_COLS | {DATE_COL}
    return [c for c in df.columns if c not in exclude]


def correlation_filter(
    df: pd.DataFrame,
    candidates: list[str],
    mi_scores: pd.Series,
    threshold: float,
) -> tuple[list[str], pd.DataFrame]:
    """
    Duyệt ma trận tương quan, với mỗi cặp vượt ngưỡng giữ cột có MI cao hơn.
    Trả về (danh sách cột được giữ, bảng log các cặp bị loại).
    """
    corr = df[candidates].corr().abs()
    removed: set[str] = set()
    log_rows: list[dict] = []

    # Duyệt tam giác trên để tránh xét cặp 2 lần
    for i in range(len(candidates)):
        col_a = candidates[i]
        if col_a in removed:
            continue
        for j in range(i + 1, len(candidates)):
            col_b = candidates[j]
            if col_b in removed:
                continue
            if corr.loc[col_a, col_b] > threshold:
                mi_a = mi_scores.get(col_a, 0.0)
                mi_b = mi_scores.get(col_b, 0.0)
                drop_col = col_b if mi_a >= mi_b else col_a
                keep_col = col_a if drop_col == col_b else col_b
                removed.add(drop_col)
                log_rows.append({
                    "Removed"     : drop_col,
                    "Kept"        : keep_col,
                    "Corr"        : round(corr.loc[col_a, col_b], 4),
                    "MI_removed"  : round(mi_scores.get(drop_col, 0.0), 6),
                    "MI_kept"     : round(mi_scores.get(keep_col, 0.0), 6),
                })

    kept = [c for c in candidates if c not in removed]
    return kept, pd.DataFrame(log_rows)


def compute_mi(df: pd.DataFrame, candidates: list[str], target: pd.Series) -> pd.Series:
    X = df[candidates].values
    y = target.values
    scores = mutual_info_regression(X, y, random_state=42)
    return pd.Series(scores, index=candidates)


def process_feature_selection(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trả về (df_out, report).
    df_out : dataset chỉ giữ feature đã chọn + giá gốc + target + Date
    report : bảng chi tiết cho từng feature
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    target = _build_target(df)

    # Bỏ hàng cuối nếu Tomorrow gold price bị NaN (hàng cuối dataset)
    valid_mask = target.notna()
    df     = df.loc[valid_mask].reset_index(drop=True)
    target = target.loc[valid_mask].reset_index(drop=True)

    candidates = _get_feature_candidates(df)
    logger.info("So feature ung vien: %d", len(candidates))

    # --- Bước 1: tính MI lần đầu để có điểm tham chiếu cho bước lọc tương quan ---
    logger.info("Tinh Mutual Information lan dau (%d features)...", len(candidates))
    mi_scores = compute_mi(df, candidates, target)

    # --- Bước 2: lọc tương quan cao ---
    logger.info("Loc tuong quan cao (nguong = %.2f)...", CORR_THRESHOLD)
    kept_after_corr, corr_log = correlation_filter(df, candidates, mi_scores, CORR_THRESHOLD)
    logger.info("Sau loc tuong quan: %d / %d feature con lai (%d bi loai)",
                len(kept_after_corr), len(candidates), len(candidates) - len(kept_after_corr))

    # --- Bước 3: tính lại MI trên tập feature còn lại ---
    logger.info("Tinh lai MI tren %d feature con lai...", len(kept_after_corr))
    mi_final = compute_mi(df, kept_after_corr, target)

    # --- Bước 4: lọc theo MI > 0 ---
    selected = mi_final[mi_final > 0].index.tolist()
    zero_mi  = mi_final[mi_final == 0].index.tolist()
    logger.info("Sau loc MI > 0: %d feature duoc giu, %d co MI = 0 bi loai",
                len(selected), len(zero_mi))

    # --- Xây dựng báo cáo ---
    # Tạo dict lý do loại cho từng cột
    removed_by_corr = set(corr_log["Removed"].tolist()) if not corr_log.empty else set()
    report_rows: list[dict] = []

    for col in candidates:
        if col in removed_by_corr:
            reason = "Tuong quan cao > " + str(CORR_THRESHOLD)
            status = "Removed"
            mi_val = round(mi_scores.get(col, 0.0), 6)
        elif col in zero_mi:
            reason = "MI = 0"
            status = "Removed"
            mi_val = 0.0
        else:
            reason = ""
            status = "Kept"
            mi_val = round(mi_final.get(col, 0.0), 6)

        report_rows.append({
            "Feature" : col,
            "MI_score": mi_val,
            "Status"  : status,
            "Reason"  : reason,
        })

    report = (
        pd.DataFrame(report_rows)
        .sort_values(["Status", "MI_score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # --- Xây dựng dataset output ---
    keep_cols = (
        [DATE_COL]
        + sorted(RAW_PRICE_COLS & set(df.columns))
        + sorted(TARGET_COLS & set(df.columns))
        + selected
    )
    df_out = df[keep_cols].copy()

    return df_out, report


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    df = pd.read_csv(INPUT_PATH)
    logger.info("Doc du lieu: %d hang x %d cot", *df.shape)

    df_out, report = process_feature_selection(df)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    logger.info("Luu du lieu: %s", OUTPUT_CSV)

    os.makedirs(os.path.dirname(REPORT_CSV), exist_ok=True)
    report.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    logger.info("Luu bao cao: %s", REPORT_CSV)

    kept    = report[report["Status"] == "Kept"]
    removed = report[report["Status"] == "Removed"]

    print("\n=== Tom tat Buoc 7 — Feature Selection ===")
    print(f"  Tong feature ung vien  : {len(report)}")
    print(f"  Feature duoc giu       : {len(kept)}")
    print(f"  Feature bi loai        : {len(removed)}")
    print(f"    - Tuong quan cao      : {(removed['Reason'].str.startswith('Tuong quan')).sum()}")
    print(f"    - MI = 0              : {(removed['Reason'] == 'MI = 0').sum()}")
    print(f"  Tong cot output        : {df_out.shape[1]}")
    print(f"  So hang output         : {len(df_out)}")
    print(f"  Bao cao luu tai        : {REPORT_CSV}")
    print(f"  Du lieu luu tai        : {OUTPUT_CSV}")

    print("\nTop 15 feature theo MI score:")
    print(kept.head(15).to_string(index=False))

    if not removed.empty:
        print(f"\nFeature bi loai ({len(removed)} cot):")
        print(removed[["Feature", "Reason", "MI_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
