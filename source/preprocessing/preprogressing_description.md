# Mô tả Pipeline Tiền Xử Lý Dữ Liệu

Tài liệu này mô tả toàn bộ 10 bước tiền xử lý dữ liệu giá vàng, thứ tự chạy, đường dẫn file, công thức và thư viện sử dụng.

---

## Tổng quan thứ tự chạy

```
converting_file_data.py        (Bước 0)
        ↓
processing_missing_value.py    (Bước 3a)
        ↓
processing_outlier.py          (Bước 3b)
        ↓
processing_time_series.py      (Bước 4)
        ↓
processing_stationarity.py     (Bước 5)
        ↓
feature_engineering.py         (Bước 6)
        ↓
feature_selection.py           (Bước 7)
        ↓
clustering.py                  (Bước 8)
        ↓
split.py                       (Bước 9)
        ↓
normalization.py               (Bước 10)
```

---

## Cây thư mục dataset

```
DTMining/dataset/
├── data_root/
│   ├── Gold Dataset (Integrated) (2).xlsx   ← file gốc duy nhất
│   └── dataset_root.csv                      ← output bước 0
├── processed/
│   ├── after_missing.csv                     ← output bước 3a
│   ├── after_outlier.csv                     ← output bước 3b
│   ├── after_time_series.csv                 ← output bước 4
│   ├── after_stationarity.csv                ← output bước 5
│   ├── after_feature_engineering.csv         ← output bước 6
│   ├── after_feature_selection.csv           ← output bước 7
│   ├── after_clustering.csv                  ← output bước 8
│   └── split/
│       ├── train.csv                         ← output bước 9
│       ├── val.csv
│       ├── test.csv
│       ├── train_scaled.csv                  ← output bước 10
│       ├── val_scaled.csv
│       ├── test_scaled.csv
│       └── walk_forward_splits.json          ← output bước 9
├── reports/
│   ├── stationarity_report.csv               ← output bước 5
│   ├── feature_selection_report.csv          ← output bước 7
│   └── clustering_report.csv                 ← output bước 8
├── models/
│   └── scaler.pkl                            ← output bước 10
└── figures/
    └── stationarity/
        ├── acf_pacf_gold_logret.png          ← output bước 5
        └── stl_gold_price.png                ← output bước 5
```

---

## Bước 0 — `converting_file_data.py`

### Mục tiêu
Chuyển đổi file Excel gốc sang định dạng CSV để các bước sau đọc dễ hơn.

### Input / Output
| | Đường dẫn |
|--|--|
| INPUT | `dataset/data_root/Gold Dataset (Integrated) (2).xlsx` |
| OUTPUT | `dataset/data_root/dataset_root.csv` |

### Thư viện
- `pandas` — đọc Excel (`pd.ExcelFile`, `pd.read_excel`) và ghi CSV

### Logic
- Đọc toàn bộ sheet trong file Excel.
- Nếu chỉ có 1 sheet → lưu thành `dataset_root.csv`.
- Nếu nhiều sheet → mỗi sheet lưu thành `dataset_root_<tên_sheet>.csv`.
- Ký tự `/` hay `\` trong tên sheet được thay bằng `-` cho an toàn tên file.

### Cách chạy
```bash
python converting_file_data.py
```

---

## Bước 3a — `processing_missing_value.py`

### Mục tiêu
Xử lý tất cả missing values và lỗi dữ liệu gốc được phát hiện qua EDA.

### Input / Output
| | Đường dẫn |
|--|--|
| INPUT | `dataset/data_root/dataset_root.csv` |
| OUTPUT | `dataset/processed/after_missing.csv` |

### Thư viện
- `pandas` — xử lý dataframe, interpolate, ffill, bfill
- `numpy` — np.nan
- `logging` — ghi log từng thao tác

### Các vấn đề xử lý & cách giải quyết

| Vấn đề | Cách xử lý |
|--------|------------|
| 2 ngày không tồn tại (`6/31/2006`, `6/31/2005`) | `pd.to_datetime(..., errors="coerce")` → NaT → xóa hàng |
| 6 ngày trùng lặp | `drop_duplicates(subset=["Date"], keep="last")` — giữ hàng cuối |
| Ký tự `-` ẩn thay NaN trong cột `U.S dollar index` | Tìm chuỗi `"-"` → thay bằng `np.nan` → ép về số |
| NaN trong cột cờ sự kiện nhị phân (0/1) | `ffill()` rồi `fillna(0)` |
| NaN trong cột giá liên tục | `interpolate(method="time", limit=3)` → `ffill()` → `bfill()` |
| Giá dầu âm tháng 4/2020 | Hợp lệ (giá tương lai thực tế) — giữ nguyên |

### Hàm chính
```
process_missing_values(df)
  → _fix_invalid_dates(df)
  → _dedup_dates(df)
  → _resolve_hidden_na(df)
  → _impute_categorical_flags(df)
  → _impute_continuous_prices(df)
```

### Cách chạy
```bash
python processing_missing_value.py
```

---

## Bước 3b — `processing_outlier.py`

### Mục tiêu
Xử lý ngoại lai bằng IQR Capping (Winsorization) — kéo giá trị vượt ngưỡng về đúng ngưỡng, không xóa hàng.

### Input / Output
| | Đường dẫn |
|--|--|
| INPUT | `dataset/processed/after_missing.csv` |
| OUTPUT | `dataset/processed/after_outlier.csv` |

### Thư viện
- `pandas` — clip, quantile

### Công thức IQR
```
Q1 = quantile(0.25)
Q3 = quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 × IQR
upper_bound = Q3 + 1.5 × IQR

Giá trị < lower_bound  →  gán bằng lower_bound
Giá trị > upper_bound  →  gán bằng upper_bound
```

### Lưu ý
- Chỉ áp dụng cho cột `float64`. Cột nhị phân 0/1 bỏ qua.
- Giá dầu âm tháng 4/2020 đã qua bước 3a, được coi là hợp lệ và giữ nguyên.

### Cách chạy
```bash
python processing_outlier.py
```

---

## Bước 4 — `processing_time_series.py`

### Mục tiêu
Làm sạch cấu trúc chuỗi thời gian: chuẩn hóa cột Date, xóa ngày cuối tuần và sắp xếp lại theo thứ tự thời gian.

### Input / Output
| | Đường dẫn |
|--|--|
| INPUT | `dataset/processed/after_outlier.csv` |
| OUTPUT | `dataset/processed/after_time_series.csv` |

### Thư viện
- `pandas` — datetime, dayofweek, sort_values

### Logic xử lý
1. `pd.to_datetime(df["Date"])` — chuẩn hóa cột Date.
2. Xóa hàng có `dayofweek >= 5` (Thứ 7 = 5, Chủ nhật = 6).
   - Lý do: giá vàng cuối tuần chỉ là copy của Thứ Sáu, nếu giữ sẽ gây spike giả trong ACF/PACF tại lag 2 mỗi tuần.
3. `sort_values("Date").reset_index(drop=True)`.
4. Log các khoảng trống ngày lễ >= 4 ngày lịch (không fill — thị trường đóng thực sự).

### Kết quả điển hình
```
8021 hàng (gốc) → 5733 hàng (chỉ ngày làm việc T2–T6)
```

### Cách chạy
```bash
python processing_time_series.py
```

---

## Bước 5 — `processing_stationarity.py`

### Mục tiêu
Kiểm định tính dừng và chuyển đổi các chuỗi không dừng sang dạng dừng để dùng trong mô hình ARIMA/SARIMAX.

### Input / Output
| | Đường dẫn |
|--|--|
| INPUT | `dataset/processed/after_time_series.csv` |
| OUTPUT CSV | `dataset/processed/after_stationarity.csv` |
| REPORT | `dataset/reports/stationarity_report.csv` |
| FIGURES | `figures/stationarity/acf_pacf_gold_logret.png`, `stl_gold_price.png` |

### Thư viện
- `statsmodels` — `adfuller`, `kpss`, `plot_acf`, `plot_pacf`, `STL`
- `matplotlib` — vẽ và lưu biểu đồ (backend `Agg` để không hiện cửa sổ)
- `numpy`, `pandas`

### Kiểm định tính dừng

| Test | Null Hypothesis | Bác bỏ H0 khi | Kết luận |
|------|----------------|---------------|----------|
| ADF | Có đơn vị gốc (không dừng) | p < 0.05 | Chuỗi dừng |
| KPSS | Chuỗi dừng | p < 0.05 | Không dừng |

Kết hợp 2 test:
- ADF bác bỏ + KPSS không bác bỏ → **Dừng**
- ADF bác bỏ + KPSS bác bỏ → **Dừng theo xu hướng**
- Còn lại → **Không dừng**

### Công thức chuyển đổi

**Log-return** (dùng cho cột giá — luôn dương):
```
{col}_logret = ln(P_t / P_{t-1})
```
Áp dụng cho: `Gold Price`, `Oil`, `S&P 500`, `DJI`, `U.S Treasury`, `U.S dollar index`, `CPI Rate`, `Silver`, `Copper`, `Iron Price`, `Platinum`, `Palladium`

**Sai phân bậc 1** (dùng cho cột lãi suất — có thể âm):
```
{col}_diff = P_t - P_{t-1}
```
Áp dụng cho: `Inflation Rate`, `Fedral Funds Rate`

> Lưu ý: `Fedral Funds Rate_diff` vẫn còn xu hướng sau sai phân do có cấu trúc break ở giai đoạn zero lower bound — chấp nhận, không xử lý thêm.

### Biểu đồ sinh ra
- **ACF/PACF** của `Gold Price_logret` → xác định bậc p (AR) và q (MA) cho ARIMA.
- **STL decomposition** của `Gold Price` gốc → tách xu hướng + mùa vụ + phần dư (period = 252 ngày giao dịch/năm, `robust=True`).

### Cách chạy
```bash
python processing_stationarity.py
```

---

## Bước 6 — `feature_engineering.py`

### Mục tiêu
Tạo các features mới từ dữ liệu đã dừng để cung cấp thông tin cho mô hình ML.

### Input / Output
| | Đường dẫn |
|--|--|
| INPUT | `dataset/processed/after_stationarity.csv` |
| OUTPUT | `dataset/processed/after_feature_engineering.csv` |

### Thư viện
- `pandas`, `logging`

### 4 nhóm feature được tạo

#### A. Lag features
- `Gold Price_logret` lag 1, 2, 3, 5, 10, 20 ngày
- 8 biến ngoại sinh (Oil, USD index, S&P 500, DJI, US Treasury, Silver, Inflation Rate, Fedral Funds Rate) lag 1, 2, 3, 5 ngày

Công thức:
```
{col}_lag{n} = df[col].shift(n)
```

#### B. Rolling statistics (trên `Gold Price_logret`)
Cửa sổ: 5, 10, 20, 60 ngày. Dùng `shift(1)` trước khi rolling để chỉ dùng thông tin đến ngày t-1 (tránh data leakage).

```python
shifted = df["Gold Price_logret"].shift(1)
roll_mean_{w} = shifted.rolling(w, min_periods=w).mean()
roll_std_{w}  = shifted.rolling(w, min_periods=w).std()
```

#### C. Cross-asset ratios (giá gốc)
```
Gold_Oil_ratio = Gold Price / Oil
Gold_USD_ratio = Gold Price / U.S dollar index
```

#### D. Calendar features
```
Month      = Date.month        (1–12)
Quarter    = Date.quarter      (1–4)
DayOfWeek  = Date.dayofweek    (0=Mon, 4=Fri)
```

### Hàng warmup bị drop
Hàng đầu có NaN do lag (tối đa lag 20) và rolling (tối đa 60 ngày) → tối đa 60 hàng đầu bị drop.

### Lưu ý quan trọng
- Feature được tạo trên **toàn bộ dataset** trước khi split (để rolling đủ dữ liệu).
- Cột target (`Tomorrow gold price`, `Next week gold price`, `Next Month gold price`) **không được dùng** để tạo feature.

### Cách chạy
```bash
python feature_engineering.py
```

---

## Bước 7 — `feature_selection.py`

### Mục tiêu
Loại bỏ features dư thừa (tương quan cao) và features không mang thông tin (MI = 0).

### Input / Output
| | Đường dẫn |
|--|--|
| INPUT | `dataset/processed/after_feature_engineering.csv` |
| OUTPUT CSV | `dataset/processed/after_feature_selection.csv` |
| REPORT | `dataset/reports/feature_selection_report.csv` |

### Thư viện
- `sklearn.feature_selection.mutual_info_regression`
- `pandas`, `numpy`, `logging`

### Quy trình 4 bước

**1. Xác định target tạm thời để tính MI:**
```
target = ln(Tomorrow gold price / Gold Price)   ← log-return 1 ngày tới
```

**2. Tính Mutual Information lần đầu** (để có điểm tham chiếu cho bước lọc tương quan):
```python
mutual_info_regression(X, y, random_state=42)
```

**3. Lọc tương quan cao (Pearson |corr| > 0.95):**
- Duyệt tam giác trên ma trận tương quan.
- Với mỗi cặp vượt ngưỡng: giữ cột có MI cao hơn, loại cột còn lại.

**4. Tính lại MI trên tập feature còn lại → loại feature có MI = 0.**

### Cột không tham gia lọc nhưng được giữ trong output
Giá gốc không dừng: `Gold Price`, `Oil`, `S&P 500`, `DJI`, `U.S Treasury`, `U.S dollar index`, `Inflation Rate`, `CPI Rate`, `Silver`, `Copper`, `Iron Price`, `Platinum`, `Palladium`, `Fedral Funds Rate` — cần thiết cho SARIMAX.

### Lưu ý
Bước này chạy trên toàn bộ dataset — khi training thực sự (bước 9+) sẽ fit lại feature selector chỉ trên tập train để tránh data leakage.

### Cách chạy
```bash
python feature_selection.py
```

---

## Bước 8 — `clustering.py`

### Mục tiêu
Gán nhãn giai đoạn thị trường (`Tang` / `Di ngang` / `Giam`) cho mỗi ngày giao dịch bằng K-Means.

### Input / Output
| | Đường dẫn |
|--|--|
| INPUT | `dataset/processed/after_feature_selection.csv` |
| OUTPUT CSV | `dataset/processed/after_clustering.csv` |
| REPORT | `dataset/reports/clustering_report.csv` |

### Thư viện
- `sklearn.cluster.KMeans`
- `sklearn.metrics` — `silhouette_score`, `davies_bouldin_score`
- `sklearn.preprocessing.StandardScaler`
- `pandas`, `numpy`, `logging`

### Quy trình

**1. Xây dựng 3 features mô tả trạng thái thị trường** từ `Gold Price_logret`:
```
logret      = Gold Price_logret (momentum tức thời)
roll_mean20 = rolling mean 20 ngày (xu hướng ngắn hạn)
roll_std20  = rolling std  20 ngày (biến động)
```
> Lưu ý: rolling ở đây KHÔNG dùng `shift(1)` vì mục đích là **mô tả trạng thái**, không phải dự báo.

**2. Chuẩn hóa bằng StandardScaler** (chỉ dùng nội bộ bước này, không lưu ra file).

**3. Fit K-Means:**
```python
KMeans(n_clusters=3, random_state=42, n_init=20)
```

**4. Đánh giá chất lượng clustering:**
- Silhouette Score — cao hơn tốt hơn (max = 1)
- Davies-Bouldin Index — thấp hơn tốt hơn (min = 0)

**5. Gán nhãn pha** bằng cách sort cluster theo `roll_mean20` trung bình:
```
Cluster có roll_mean20 thấp nhất  → "Giam"
Cluster có roll_mean20 ở giữa    → "Di ngang"
Cluster có roll_mean20 cao nhất  → "Tang"
```

**6. Thêm cột `market_phase`** vào dataset. Dùng `ffill()` để lấp 19 hàng đầu chưa đủ 20 điểm rolling.

### Cách chạy
```bash
python clustering.py
```

---

## Bước 9 — `split.py`

### Mục tiêu
Chia dataset thành 3 tập theo thời gian và sinh ra các fold walk-forward validation.

### Input / Output
| | Đường dẫn |
|--|--|
| INPUT | `dataset/processed/after_clustering.csv` |
| OUTPUT DIR | `dataset/processed/split/` |
| OUTPUT | `train.csv`, `val.csv`, `test.csv`, `walk_forward_splits.json` |

### Thư viện
- `pandas`, `json`, `logging`

### Mốc chia tập (cứng theo ngày)

| Tập | Khoảng thời gian | Tỷ lệ |
|-----|-----------------|-------|
| Train | 2001-03-27 → 2017-12-31 | ~80% |
| Validation | 2018-01-01 → 2019-12-31 | ~9% |
| Test | 2020-01-01 → 2022-12-30 | ~11% |

> Test bắt đầu từ 2020 để bao gồm giai đoạn COVID và biến động lớn của giá vàng.

### Walk-forward validation (Expanding Window)
Chỉ trên tập Train:
- Train tối thiểu 3 năm đầu, mỗi fold mở rộng thêm 1 năm.
- Val window = 1 năm.
- Kết quả lưu ra `walk_forward_splits.json` để dùng lại khi training.

### Kiểm tra data leakage
Đảm bảo không có cột target nào nằm trong feature set trước khi lưu file.

### Lưu ý về data leakage (đã được xử lý ở các bước trước)
- Lag features dùng `shift(n>=1)` → không nhìn hiện tại.
- Rolling features dùng `shift(1)` trước rolling → an toàn.
- `market_phase` tính từ dữ liệu đã có, không dùng thông tin tương lai.
- Cross-asset ratios dùng giá cùng ngày (available at end-of-day).

### Cách chạy
```bash
python split.py
```

---

## Bước 10 — `normalization.py`

### Mục tiêu
Chuẩn hóa các feature bằng StandardScaler, fit chỉ trên tập train để tránh data leakage.

### Input / Output
| | Đường dẫn |
|--|--|
| INPUT | `dataset/processed/split/train.csv`, `val.csv`, `test.csv` |
| INPUT REPORT | `dataset/reports/feature_selection_report.csv` |
| OUTPUT | `dataset/processed/split/train_scaled.csv`, `val_scaled.csv`, `test_scaled.csv` |
| SCALER | `dataset/models/scaler.pkl` |

### Thư viện
- `sklearn.preprocessing.StandardScaler`
- `pandas`, `pickle`, `logging`

### Quy trình

**1. Lấy danh sách feature cần scale** từ `feature_selection_report.csv` (Status == "Kept"), lọc bỏ cột không được scale.

**Cột KHÔNG scale:**
```
Date, market_phase,
Tomorrow gold price, Next week gold price, Next Month gold price,
Gold Price, Oil, S&P 500, DJI, U.S Treasury, U.S dollar index,
Inflation Rate, CPI Rate, Silver, Copper, Iron Price,
Platinum, Palladium, Fedral Funds Rate
```

**2. Fit StandardScaler CHỈ trên train:**
```python
scaler.fit(train[feature_cols])
```

**Công thức StandardScaler:**
```
X_scaled = (X - mean_train) / std_train
```

**3. Transform cả 3 tập bằng scaler đã fit:**
```python
train_scaled = scaler.transform(train[feature_cols])
val_scaled   = scaler.transform(val[feature_cols])    # dùng mean/std của train
test_scaled  = scaler.transform(test[feature_cols])   # dùng mean/std của train
```

**4. Lưu scaler ra `scaler.pkl`** để dùng lại khi inference.

### Kiểm tra sau scale (trên train)
- `max |mean|` ≈ 0 (xấp xỉ 0)
- `std` min/max ≈ 1

### Cách chạy
```bash
python normalization.py
```

---

## Hướng dẫn chạy toàn bộ pipeline

Chạy tuần tự từng file theo thứ tự:

```bash
cd D:\University\Data Mining\Project_DTMining\DTMining\source\preprocessing

python converting_file_data.py
python processing_missing_value.py
python processing_outlier.py
python processing_time_series.py
python processing_stationarity.py
python feature_engineering.py
python feature_selection.py
python clustering.py
python split.py
python normalization.py
```

> Mỗi bước in ra tóm tắt kết quả khi chạy xong. Nếu một bước thất bại, kiểm tra file input của bước đó có tồn tại chưa (output của bước trước).

---

## Thư viện cần cài

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib
```

---

## Quy ước đặt tên cột

| Hậu tố | Ý nghĩa |
|--------|---------|
| `_logret` | Log-return: `ln(P_t / P_{t-1})` |
| `_diff` | Sai phân bậc 1: `P_t - P_{t-1}` |
| `_lag{n}` | Giá trị lag n ngày: `shift(n)` |
| `_roll_mean{w}` | Rolling mean cửa sổ w ngày |
| `_roll_std{w}` | Rolling std cửa sổ w ngày |
| `_ratio` | Tỷ lệ giữa 2 giá gốc |
