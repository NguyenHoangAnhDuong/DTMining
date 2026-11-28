# Thông tin dự án

## Bài toán

Phân tích giai đoạn thị trường và dự đoán giá vàng dựa trên dữ liệu giá vàng lịch sử, các chỉ số kinh tế, tài chính và sự kiện toàn cầu.

Câu hỏi chính của dự án: **phân lớp giai đoạn thị trường (K-Means) trước khi dự đoán có giúp dự báo giá vàng tốt hơn không?**

## Tập dữ liệu

- **EGPBD** – An Event-based Gold Price Benchmark Dataset – 8.030 bản ghi, 36 biến, 01/2001–12/2022 (giá vàng, chỉ số kinh tế/tài chính, giá hàng hóa, sự kiện toàn cầu).
- Link: <https://data.mendeley.com/datasets/y6vj5kzfpz/1>

## Chạy toàn bộ dự án

```bash
python -m source.run_pipeline                # tiền xử lý → K-Means → train → báo cáo
python -m source.run_pipeline --from model   # chỉ train lại model + báo cáo
```

Mọi đường dẫn và tham số chung nằm ở một nơi: `source/preprocessing_v3/config.py`.

## Nguyên tắc phân chia trách nhiệm

| Lớp | Thư mục | Làm gì | Không làm gì |
|---|---|---|---|
| **Dữ liệu** | `source/preprocessing_v3/` | Làm sạch, chống nhiễu, biến đổi dừng (`_logret`/`_diff`), định nghĩa target, chia tập, gán nhãn K-Means | Không tạo feature, không chuẩn hóa, không chọn feature |
| **Hạ tầng chung cho model** | `source/model/common/` | Đọc 2 nhánh, purge chống rò rỉ, quy đổi log-return → giá, chỉ số đánh giá, khoảng dự đoán, định dạng kết quả | Không tạo feature |
| **Từng model** | `source/model/<model>/` | `features.py`: feature riêng của model · `train.py`: tham số, tuning, huấn luyện, dự đoán | — |

- `_logret`/`_diff` thuộc lớp dữ liệu vì là bước kiểm định & biến đổi tính dừng (step05), và K-Means cũng cần.
- Target đặt ở lớp dữ liệu (step06) để **mọi model dự đoán cùng một đại lượng** → so sánh được với nhau.
- Feature (lag, rolling, chuẩn hóa, cửa sổ chuỗi, biến ngoại sinh, ...) **chỉ được tính trong bước model**, trong bộ nhớ, không ghi ra `dataset/`.

## Luồng tổng thể

```text
Dữ liệu gốc
    │
    ▼
[1] Tiền xử lý chung (đơn giản, chỉ chống nhiễu)  ──►  base_dataset.csv
    │
    ▼
[2] Chia Train / Val / Test theo thời gian (cùng mốc cho 2 nhánh)
    ├──────────────────────────────┐
    ▼                              ▼
Nhánh KHÔNG phân lớp           Nhánh CÓ phân lớp
*_base.csv                     K-Means (fit trên Train) gán giai đoạn → *_kmeans.csv + kmeans.joblib
    │                              │
    ▼                              ▼
[3] Bước model — mỗi model tự tính feature + tham số của mình
    (trong cùng model: 2 nhánh cùng feature gốc, nhánh kmeans thêm nhãn giai đoạn)
    → train × h = 1, 7, 30
    │                              │
    └──────────────┬───────────────┘
                   ▼
[4] So sánh: dự đoán điểm + khoảng dự đoán, kiểm định Diebold-Mariano
```

## Bước 1 – Tiền xử lý chung

Chỉ xử lý ở mức đơn giản để loại bỏ dữ liệu lỗi và tín hiệu nhiễu. **Không** tạo feature, **không** chuẩn hóa ở bước này.

| Step | File | Nội dung |
|------|------|----------|
| 01 | `step01_cleaning.py` | Cách ly ngày không hợp lệ, xử lý bản ghi trùng/xung đột, chuyển kiểu số |
| 02 | `step02_time_series.py` | Loại bỏ ngày cuối tuần (trước khi điền thiếu) |
| 03 | `step03_missing.py` | Nội suy theo thời gian cho giá, forward-fill cho biến sự kiện |
| 04 | `step04_outlier.py` | Phát hiện outlier — chỉ báo cáo, không cắt |
| 05 | `step05_stationarity.py` | Kiểm định ADF/KPSS, thêm cột `_logret`/`_diff`, ACF/PACF, phân rã STL |
| 06 | `step06_targets.py` | Tạo target đa khung dự báo h = 1, 7, 30 |

Output: `dataset/processed_v3/base_dataset.csv`, báo cáo trong `dataset/processed_v3/reports/`, hình trong `figures/v3/`.

## Bước 2 – Chia dữ liệu và phân lớp giai đoạn thị trường với K-Means

- Chia theo thời gian: Train 2001–2017, Val 2018–2019, Test 2020–2022 (`step07_split.py`).
- K-Means (k = 4) trên 2 trục **xu hướng 60 phiên** và **biến động 20 phiên** (chỉ dùng quá khứ), chỉ fit trên Train (`step09_clustering.py`):
  **Giảm (Bear) – Tăng (Bull) – Tăng mạnh (StrongBull) – Biến động mạnh (Volatile)**.
- Output (`dataset/processed_v3/`):
  - Nhánh không phân lớp: `split/train_base.csv`, `val_base.csv`, `test_base.csv`
  - Nhánh có phân lớp: `split/train_kmeans.csv`, `val_kmeans.csv`, `test_kmeans.csv` (thêm `kmeans_<Regime>`, `regime_id`, `market_phase`)
  - Model K-Means: `models/kmeans.joblib`; `split/walk_forward_splits.json`
- Báo cáo: **`results/kmeans/result_k_mean.md`** — chất lượng cụm, lý do chọn feature và k, thống kê từng giai đoạn, hình minh họa.

## Bước 3 – Bước model: feature, tham số và huấn luyện

### Hạ tầng chung (`source/model/common/`) — không tạo feature

| File | Nội dung |
|---|---|
| `data.py` | Đọc 2 nhánh (`read_branch`), target chuẩn (log-return h phiên), purge ranh giới Train/Val (`usable_mask`, `purge_fold`), quy đổi giá (`to_price`), định dạng `predictions.csv` |
| `metrics.py` | MAE, RMSE, MAPE, R², MASE, DA; `metrics.csv` kèm naive; kiểm định Diebold-Mariano |
| `intervals.py` | Khoảng dự đoán conformal: độ rộng chung (base) hoặc theo giai đoạn (kmeans) |

### XGBoost (`source/model/xgboost_model/`)

- `features.py` — feature riêng cho model cây: chỉ đại lượng dừng (log-return/sai phân, trễ & rolling log-return vàng, khoảng cách MA50/MA200, cờ & đếm sự kiện, lịch); không chuẩn hóa (cây không cần); không dùng giá tuyệt đối (cây không ngoại suy được khi giá Test vượt đỉnh Train). Nhánh kmeans thêm nhãn giai đoạn.
- `train.py` — Optuna tuning trên walk-forward (có purge), early stopping theo Val, refit Train+Val trước khi đánh giá Test; khoảng dự đoán hiệu chỉnh trên sai số out-of-fold 2004–2017; `predict_latest()` cho web app.
- Output theo `docs/result_contract.md`: `results/forecasting/xgboost/` (`metrics.csv`, `predictions.csv`, `feature_importance.csv`, `metadata.json`, `models/`).

### Thêm model mới (SARIMAX, LightGBM, Prophet, LSTM, ...)

1. Tạo `source/model/<model>/features.py`: tính feature theo nhu cầu của model (vd. LSTM: chuẩn hóa fit trên Train + cửa sổ chuỗi; SARIMAX: biến ngoại sinh; LightGBM: có thể dùng lại `xgboost_model/features.py`).
2. Tạo `source/model/<model>/train.py` có hàm `run()`: dùng `common/data.py` để lấy dữ liệu 2 nhánh và target, `common/metrics.py` + `common/intervals.py` để đánh giá, ghi kết quả vào `results/forecasting/<model>/`.
3. Đăng ký vào `MODELS` trong `source/run_pipeline.py` — báo cáo so sánh tự nhận model mới.

## Bước 4 – Báo cáo so sánh

**`results/forecasting/result_forecast.md`**: so sánh có/không phân lớp cho từng khoảng dự đoán — MAE, RMSE, MAPE, R², MASE, DA, so với naive; kiểm định Diebold-Mariano; khoảng dự đoán (coverage, độ rộng, Winkler); kết quả theo từng giai đoạn; feature importance.

## Ứng dụng web

1. **Chọn có phân lớp trước khi dự đoán hay không.**
2. **Chọn mô hình** dự đoán chuỗi thời gian.
3. **Chọn khoảng dự đoán** (h = 1, 7, 30 ngày).
4. **Dự đoán** — `source.model.xgboost_model.train.predict_latest(history, branch, horizon)`:
   - *Không phân lớp:* tạo feature → gọi model nhánh `base`.
   - *Có phân lớp:* **gọi model K-Means trước** (`assign_regime`) để xác định giai đoạn hiện tại → tạo feature → gọi model nhánh `kmeans`; khoảng tin cậy theo giai đoạn.
5. **Hiển thị:** giá dự đoán, khoảng tin cậy 80%/95%, giai đoạn thị trường hiện tại (nếu có phân lớp).
6. **So sánh:** kết quả có phân lớp và không phân lớp của cùng một model, lấy từ `results/forecasting/`.

## Chỉ số đánh giá

- **Dự đoán điểm:** MAE, RMSE, MAPE, R², MASE (so với naive), Directional Accuracy; kiểm định Diebold-Mariano.
- **Khoảng dự đoán:** Coverage, độ rộng trung bình, Winkler score.
- **Phân cụm:** Silhouette, Davies-Bouldin, Calinski-Harabasz.

## Cấu trúc thư mục

```text
source/
  run_pipeline.py            # chạy toàn bộ
  preprocessing_v3/          # config, step01–07, step09, run_stage1/2, report_kmeans
  model/
    common/                  # data, metrics, intervals — dùng chung, không tạo feature
    xgboost_model/           # features.py + train.py riêng của XGBoost
    report_forecast.py       # so sánh có/không phân lớp cho mọi model
  services/                  # đọc dữ liệu / kết quả cho web app
dataset/processed_v3/        # base_dataset, split/, models/kmeans.joblib, reports/
results/kmeans/              # result_k_mean.md + figures/
results/forecasting/         # result_forecast.md + figures/ + <model>/
pages/, components/, app.py  # web app (Streamlit)
```
