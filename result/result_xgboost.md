# Kết quả mô hình XGBoost — Dự báo giá vàng ngày mai

> Ngày chạy: 2026-11-07 18:52  
> XGBoost 3.4.1

---

## 1. Cấu hình mô hình

| Tham số | Giá trị |
|---|---|
| Target | `Tomorrow gold price` |
| Số feature | 70 |
| Số estimators tốt nhất | 63 |
| Learning rate | 0.14848 |
| Max depth | 8 |
| Min child weight | 8 |
| Subsample | 0.6692 |
| Colsample bytree | 0.9626 |
| Reg alpha (L1) | 2.773191 |
| Reg lambda (L2) | 0.918177 |
| Hyperparameter tuning | Optuna TPE — 50 trials, 3 WF folds cuối |

---

## 2. Kết quả Walk-Forward Cross-Validation (trên tập Train 2001–2017, 14 folds)

| Fold | MAE | RMSE | MAPE (%) | R² | MASE | Directional Accuracy (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.33 | 16.03 | 2.95 | 0.2462 | 4.0427 | 51.7 |
| 2 | 16.89 | 27.01 | 3.56 | 0.0046 | 6.6479 | 48.3 |
| 3 | 88.79 | 96.99 | 14.35 | -5.0283 | 13.4622 | 45.3 |
| 4 | 40.15 | 65.02 | 5.24 | -0.2141 | 7.5442 | 52.7 |
| 5 | 78.04 | 88.73 | 8.70 | -0.7334 | 6.5506 | 49.0 |
| 6 | 45.43 | 76.88 | 4.22 | 0.1979 | 4.8258 | 55.8 |
| 7 | 71.03 | 103.26 | 5.38 | -0.0949 | 8.0104 | 49.6 |
| 8 | 189.78 | 238.82 | 11.33 | -1.6052 | 12.7214 | 44.0 |
| 9 | 12.07 | 15.09 | 0.72 | 0.9439 | 1.0419 | 66.5 |
| 10 | 21.12 | 25.93 | 1.56 | 0.9692 | 1.6241 | 62.2 |
| 11 | 5.26 | 6.93 | 0.42 | 0.9813 | 0.6786 | 72.1 |
| 12 | 5.42 | 7.15 | 0.47 | 0.9842 | 0.6816 | 74.6 |
| 13 | 3.98 | 5.26 | 0.32 | 0.9950 | 0.4622 | 78.5 |
| 14 | 2.76 | 3.85 | 0.22 | 0.9881 | 0.4457 | 78.7 |
| **TB** | **42.36** | **55.50** | **4.24** | **-0.0975** | **4.9099** | **59.2** |

---

## 3. Kết quả trên tập Validation (2018–2019)

| Metric | Giá trị |
|---|---|
| MAE | 4.0823 USD/oz |
| RMSE | 6.3999 USD/oz |
| MAPE | 0.2964% |
| R² | 0.9957 |
| MASE | 0.6333 |
| Directional Accuracy | 77.74% |

---

## 4. Kết quả trên tập Test (2020–2022)

| Metric | Giá trị |
|---|---|
| MAE | 54.8318 USD/oz |
| RMSE | 79.1231 USD/oz |
| MAPE | 2.9217% |
| R² | 0.3946 |
| MASE | 4.0581 |
| Directional Accuracy | 54.87% |

---

## 5. Top 10 Feature Importance (gain)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `Gold Price` | 0.936779 |
| 2 | `CPI Rate` | 0.049598 |
| 3 | `Gold_USD_ratio` | 0.005543 |
| 4 | `Silver` | 0.004681 |
| 5 | `Platinum` | 0.000653 |
| 6 | `Copper` | 0.000495 |
| 7 | `Gold_logret_roll_mean60` | 0.000446 |
| 8 | `Gold Price_logret` | 0.000292 |
| 9 | `DJI` | 0.000238 |
| 10 | `Silver_logret` | 0.000106 |

---

## 6. Chú giải metric

| Metric | Mô tả |
|---|---|
| **MAE** | Sai số tuyệt đối trung bình (USD/oz) — đơn vị giá vàng |
| **RMSE** | Căn bậc hai sai số bình phương trung bình (USD/oz) — phạt sai số lớn nhiều hơn |
| **MAPE** | Sai số phần trăm tuyệt đối trung bình (%) |
| **R²** | Hệ số xác định — càng gần 1 mô hình giải thích variance càng tốt |
| **MASE** | MAE chuẩn hóa bởi naive random-walk; **MASE < 1** có nghĩa tốt hơn baseline |
| **DA** | Directional Accuracy — % ngày dự báo đúng chiều tăng/giảm so với giá thực ngày trước |

- **Naive baseline**: dự báo ngày mai = giá đóng cửa hôm nay (random walk)
- **Tập test 2020–2022**: bao gồm giai đoạn COVID-19 và chu kỳ tăng mạnh của giá vàng
- **Model**: `source/model/xgboost_model.pkl`
