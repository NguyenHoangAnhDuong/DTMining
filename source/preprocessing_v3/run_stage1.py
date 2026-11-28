"""
Runner — Stage 1: Data Cleaning → Base Dataset

Chạy từ thư mục gốc dự án (DTMining/):
    python -m source.preprocessing_v3.run_stage1

Output: dataset/processed_v3/base_dataset.csv
  Chứa: dữ liệu đã làm sạch + cột _logret/_diff + cột target h=1,7,30
  Chưa có: lag/rolling feature, cluster columns, normalization
  → Sang Stage 2 để bổ sung feature và chuẩn bị cho từng model
"""
import logging
import sys
from pathlib import Path

import pandas as pd

# Đảm bảo project root trong sys.path khi chạy trực tiếp
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source.preprocessing_v3.config import BASE_DATASET, OUT_DIR, RAW_DATA
import source.preprocessing_v3.step01_cleaning     as s01
import source.preprocessing_v3.step02_time_series  as s02
import source.preprocessing_v3.step03_missing      as s03
import source.preprocessing_v3.step04_outlier      as s04
import source.preprocessing_v3.step05_stationarity as s05
import source.preprocessing_v3.step06_targets      as s06


def run() -> pd.DataFrame:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("STAGE 1 — Data Cleaning → Base Dataset")
    logger.info("Input : %s", RAW_DATA)
    logger.info("=" * 60)

    df = pd.read_csv(RAW_DATA)
    logger.info("Loaded raw data: %d rows × %d cols", *df.shape)

    df = s01.run(df)   # date quarantine, duplicate/conflict, numeric convert
    df = s02.run(df)   # remove weekends (TRƯỚC khi impute)
    df = s03.run(df)   # time-interpolation prices, ffill events
    df = s04.run(df)   # outlier detection — report only, NO capping
    df = s05.run(df)   # ADF+KPSS, add _logret/_diff, ACF/PACF, STL
    df = s06.run(df)   # multi-horizon targets h=1,7,30

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(BASE_DATASET, index=False)

    logger.info("=" * 60)
    logger.info("STAGE 1 COMPLETE")
    logger.info("Base dataset: %d rows × %d cols", *df.shape)
    logger.info("Saved → %s", BASE_DATASET)
    logger.info("=" * 60)
    return df


if __name__ == "__main__":
    run()
