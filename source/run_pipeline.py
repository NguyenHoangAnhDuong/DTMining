"""
Chạy toàn bộ pipeline của dự án theo đúng thứ tự.

Chạy từ thư mục gốc dự án (DTMining/):
    python -m source.run_pipeline              # chạy tất cả
    python -m source.run_pipeline --from model # bỏ qua tiền xử lý, chỉ train + báo cáo

Các bước:
  1. preprocess : Stage 1 — làm sạch, chống nhiễu → base_dataset.csv
  2. split      : Stage 2 — tách 2 nhánh (base / kmeans), train K-Means
  3. kmeans     : Báo cáo K-Means → results/kmeans/result_k_mean.md
  4. model      : Train từng model trong MODELS cho 2 nhánh × h = 1, 7, 30 → results/forecasting/<model>/
  5. report     : So sánh có/không phân lớp → results/forecasting/result_forecast.md

Thêm model mới: tạo source/model/<tên>/ (features.py + train.py có hàm run()) rồi thêm vào MODELS.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.model import report_forecast
from source.model.xgboost_model import train as xgboost_train
from source.preprocessing_v3 import report_kmeans, run_stage1, run_stage2

# Mỗi model tự lo feature + tham số; chỉ cần đăng ký hàm train ở đây
MODELS = {
    "xgboost": xgboost_train.run,
}


def _train_all_models():
    for name, train in MODELS.items():
        print(f"--- {name} ---")
        train()


STEPS = {
    "preprocess": run_stage1.run,
    "split":      run_stage2.run,
    "kmeans":     report_kmeans.run,
    "model":      _train_all_models,
    "report":     report_forecast.run,
}


def main():
    parser = argparse.ArgumentParser(description="DTMining pipeline")
    parser.add_argument("--from", dest="start", choices=STEPS, default="preprocess",
                        help="bước bắt đầu (mặc định: preprocess)")
    args = parser.parse_args()
    names = list(STEPS)
    for name in names[names.index(args.start):]:
        print(f"\n===== {name.upper()} =====")
        STEPS[name]()


if __name__ == "__main__":
    main()
