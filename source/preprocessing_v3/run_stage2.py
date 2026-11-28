"""
Runner — Stage 2: Tách 2 nhánh dữ liệu (KHÔNG phân lớp / CÓ phân lớp K-Means)

Yêu cầu: Stage 1 đã chạy xong (base_dataset.csv tồn tại).

Chạy từ thư mục gốc dự án (DTMining/):
    python -m source.preprocessing_v3.run_stage2

Các bước:
  07 → Temporal split base_dataset (cùng mốc ngày cho cả 2 nhánh)
       → Nhánh A (không phân lớp): train_base / val_base / test_base
  09 → Thêm feature clustering (rolling 20 phiên, chỉ dùng quá khứ) trên toàn bộ data,
       split theo cùng mốc ngày, fit K-Means CHỈ trên Train, gán nhãn cho cả 3 tập
       → Nhánh B (có phân lớp): train_kmeans / val_kmeans / test_kmeans + kmeans.joblib

KHÔNG làm ở stage này: feature engineering cho dự đoán nằm ở source/model/dataset.py,
áp dụng giống hệt nhau cho cả 2 nhánh.

Output:
  dataset/processed_v3/split/
    train_base.csv,   val_base.csv,   test_base.csv     — nhánh A
    train_kmeans.csv, val_kmeans.csv, test_kmeans.csv   — nhánh B
    walk_forward_splits.json                            — fold info cho CV
  dataset/processed_v3/models/kmeans.joblib             — model K-Means cho web app
  dataset/processed_v3/reports/clustering_metrics.csv   — Silhouette, Davies-Bouldin
"""
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source.preprocessing_v3.config import BASE_DATASET, KMEANS_SPLIT_SUFFIX, SPLIT_DIR
import source.preprocessing_v3.step07_split      as s07
import source.preprocessing_v3.step09_clustering as s09


def run():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("STAGE 2 — Split 2 nhánh: không phân lớp / K-Means")
    logger.info("Input: %s", BASE_DATASET)
    logger.info("=" * 60)

    if not BASE_DATASET.exists():
        raise FileNotFoundError(
            f"base_dataset.csv not found: {BASE_DATASET}\n"
            "Run Stage 1 first: python -m source.preprocessing_v3.run_stage1"
        )

    df = pd.read_csv(BASE_DATASET, parse_dates=["Date"])
    logger.info("Loaded base dataset: %d rows × %d cols", *df.shape)

    # ── Nhánh A: không phân lớp ────────────────────────────────────────────
    train_a, val_a, test_a, _wf = s07.run(df)   # lưu *_base.csv + walk_forward_splits.json

    # ── Nhánh B: có phân lớp K-Means ───────────────────────────────────────
    # Rolling chỉ dùng quá khứ → tính trên toàn bộ data rồi mới split (giữ lịch sử qua ranh giới)
    df_cl = s09.add_cluster_features(df)
    train_cl, val_cl, test_cl = s07.temporal_split(df_cl)
    train_b, val_b, test_b, _bundle = s09.run(train_cl, val_cl, test_cl)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train_b), ("val", val_b), ("test", test_b)]:
        part.to_csv(SPLIT_DIR / f"{name}_{KMEANS_SPLIT_SUFFIX}.csv", index=False)

    for name, part in [("Train", train_b), ("Val", val_b), ("Test", test_b)]:
        dist = part["market_phase"].value_counts().to_dict()
        logger.info("  %-5s regime distribution: %s", name, dist)

    logger.info("=" * 60)
    logger.info("STAGE 2 COMPLETE")
    logger.info("Nhánh A (base)  : Train %d | Val %d | Test %d | %d cols",
                len(train_a), len(val_a), len(test_a), train_a.shape[1])
    logger.info("Nhánh B (kmeans): Train %d | Val %d | Test %d | %d cols",
                len(train_b), len(val_b), len(test_b), train_b.shape[1])
    logger.info("Output → %s", SPLIT_DIR)
    logger.info("=" * 60)

    return (train_a, val_a, test_a), (train_b, val_b, test_b)


if __name__ == "__main__":
    run()
