"""
Runner — Stage 2: Feature Engineering → Model-Ready Splits

Yêu cầu: Stage 1 đã chạy xong (base_dataset.csv tồn tại).

Chạy từ thư mục gốc dự án (DTMining/):
    python -m source.preprocessing_v3.run_stage2

Thứ tự thực hiện (tránh data leakage hoàn toàn):
  07 → Temporal split (TRƯỚC khi fit bất cứ thứ gì)
  08 → Feature engineering trên toàn bộ data (lag/rolling an toàn vì chỉ dùng quá khứ)
       → Re-split theo mốc ngày
  09 → Clustering: fit K-Means + GMM CHỈ trên Train, transform cả 3 tập
  10 → Feature selection: tính MI + corr filter CHỈ trên Train
  11 → Normalization: fit StandardScaler CHỈ trên Train

Output (dataset/processed_v3/split/):
  train.csv / val.csv / test.csv           — đã có feature, chưa scale
  train_scaled.csv / val_scaled.csv / test_scaled.csv — đã scale
  walk_forward_splits.json                 — fold info cho CV
"""
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source.preprocessing_v3.config import BASE_DATASET, SPLIT_DIR
import source.preprocessing_v3.step07_split              as s07
import source.preprocessing_v3.step08_feature_engineering as s08
import source.preprocessing_v3.step09_clustering          as s09
import source.preprocessing_v3.step10_feature_selection   as s10
import source.preprocessing_v3.step11_normalization       as s11


def run():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("STAGE 2 — Feature Engineering → Model-Ready Splits")
    logger.info("Input: %s", BASE_DATASET)
    logger.info("=" * 60)

    if not BASE_DATASET.exists():
        raise FileNotFoundError(
            f"base_dataset.csv not found: {BASE_DATASET}\n"
            "Run Stage 1 first: python -m source.preprocessing_v3.run_stage1"
        )

    df = pd.read_csv(BASE_DATASET, parse_dates=["Date"])
    logger.info("Loaded base dataset: %d rows × %d cols", *df.shape)

    # 07 — split trước khi fit bất cứ thứ gì
    train_base, val_base, test_base, wf_folds = s07.run(df)

    # 08 — feature engineering trên toàn bộ data, sau đó re-split theo Date
    # (lag/rolling cần lịch sử xuyên suốt → không tách trước)
    df_fe = s08.run(df, drop_warmup=True)

    t_end = train_base["Date"].max()
    v_start, v_end = val_base["Date"].min(), val_base["Date"].max()
    x_start = test_base["Date"].min()

    train_fe = df_fe[df_fe["Date"] <= t_end].reset_index(drop=True)
    val_fe   = df_fe[(df_fe["Date"] >= v_start) & (df_fe["Date"] <= v_end)].reset_index(drop=True)
    test_fe  = df_fe[df_fe["Date"] >= x_start].reset_index(drop=True)

    logger.info("After FE re-split — Train: %d  Val: %d  Test: %d",
                len(train_fe), len(val_fe), len(test_fe))

    # 09 — clustering fit on train
    train_cl, val_cl, test_cl, _bundle = s09.run(train_fe, val_fe, test_fe)

    # 10 — feature selection fit on train
    train_fs, val_fs, test_fs, sel_config = s10.run(train_cl, val_cl, test_cl)

    # 11 — normalization fit on train
    train_sc, val_sc, test_sc, _scaler = s11.run(train_fs, val_fs, test_fs, sel_config)

    # Save
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train_fs.to_csv(SPLIT_DIR / "train.csv",        index=False)
    val_fs.to_csv(SPLIT_DIR   / "val.csv",          index=False)
    test_fs.to_csv(SPLIT_DIR  / "test.csv",         index=False)
    train_sc.to_csv(SPLIT_DIR / "train_scaled.csv", index=False)
    val_sc.to_csv(SPLIT_DIR   / "val_scaled.csv",   index=False)
    test_sc.to_csv(SPLIT_DIR  / "test_scaled.csv",  index=False)

    logger.info("=" * 60)
    logger.info("STAGE 2 COMPLETE")
    logger.info("Train: %d rows  Val: %d rows  Test: %d rows", len(train_sc), len(val_sc), len(test_sc))
    logger.info("Cols: %d (unscaled) | %d (scaled)", train_fs.shape[1], train_sc.shape[1])
    logger.info("Output → %s", SPLIT_DIR)
    logger.info("=" * 60)

    return train_sc, val_sc, test_sc


if __name__ == "__main__":
    run()
