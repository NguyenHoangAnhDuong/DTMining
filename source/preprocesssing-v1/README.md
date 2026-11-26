# Gold Price Preprocessing


## Cài đặt và chạy (Python 3.11/3.12)
Đặt Excel gốc cạnh main.py hoặc chỉ định đường dẫn tuyệt đối. Không cần sửa đường dẫn trong source.

```bash
pip install -r requirements.txt
python main.py --input "Gold Dataset (Integrated) (2).xlsx" --output processed/market --folds 5
python main.py --input "Gold Dataset (Integrated) (2).xlsx" --output processed/extended --extended --folds 5
python main.py --input "Gold Dataset (Integrated) (2).xlsx" --output processed/market_2d --cluster-dimensions 2 --folds 5
```

Mỗi lần chạy dùng thư mục mới hoặc rỗng. Có thể dùng `--folds 2` để chạy thử nhanh. Các báo cáo và hình được sinh tự động. Không cần chạy từng module.

## Đọc source theo thứ tự

| Module | Kỹ thuật và minh chứng |
|---|---|
| data_loader | Kiểm tra schema, đọc Excel/CSV |
| data_understanding | EDA, dtype, missing, unique, describe, biểu đồ missing |
| data_cleaning | Parse ngày, numeric, duplicate, cách ly mâu thuẫn |
| time_series | Sort, loại cuối tuần, báo cáo gap, biểu đồ giá |
| missing_value | Forward fill tối đa 5, không điền target hoặc event |
| outlier_detection | IQR và boxplot trên lịch sử train, không capping |
| stationarity | ADF/KPSS trước và sau biến đổi 14 chuỗi |
| time_series_analysis | ACF/PACF, STL, bảng và hình |
| feature_engineering | Return, lag, rolling, ratio, calendar, macro/event |
| target_engineering | Giá và log-return tương lai, ngày target |
| split | Chia thời gian, purge |
| feature_selection | MI, Pearson, lý do giữ/loại, top-k |
| normalization | StandardScaler fit train |
| clustering | K-Means, GMM, nhãn, xác suất, đánh giá và thử K=2..6 |
| walk_forward | Expanding window, purge lại mỗi fold |
| data_validation | Kiểm tra ranh giới, missing, infinity, xác suất |

## Cấu trúc đầu ra
- `intermediate/`: dữ liệu từng giai đoạn, có thể chứa NaN do warmup; không dùng trực tiếp để train.
- `base_dataset.csv`: dữ liệu nghiên cứu có nhiều target và cột gốc, không dùng toàn bộ làm X.
- `reports/`, `figures/`: minh chứng EDA, IQR, stationarity, STL, ACF/PACF.
- `horizons/h1`, `h7`, `h30`: train/val/test, features.json, MI, scaler, clustering, audit, hình.
- `walk_forward/h*/fold_*`: train/val và artifacts fit riêng từng fold.

## Ví dụ đọc dữ liệu cho XGBoost/LightGBM
```python
import json
from pathlib import Path
import pandas as pd

path = Path('processed/market/horizons/h7')
config = json.loads((path / 'features.json').read_text(encoding='utf8'))
features = config['selection']['price']['40'] + config['task_additions']['task1_kmeans']
target = config['targets']['price']
train = pd.read_csv(path / 'train.csv')
X_train = train[features]
y_train = train[target]
```
Dùng config riêng từng fold khi tuning. CSV chưa scale. Scaler lưu sẵn chỉ áp dụng đúng toàn bộ base features theo thứ tự config all_base; nếu dùng subset/cluster thì fit scaler mới trên train tương ứng. Không load joblib không rõ nguồn. Với SARIMAX/Prophet/Holt-Winters cần lịch sử giá liên tục và giao thức forecast riêng, không mặc định dùng bảng supervised đã purge. LSTM cần sequence chỉ sử dụng quá khứ. Naive dự báo P_t. Test chỉ dùng khi đã chốt mô hình.

## Giới hạn và quyết định thiết kế
- Market-only 98 feature; Extended 140 feature. Extended lag1 không đảm bảo thời điểm công bố thật; cần dữ liệu point-in-time và kiểm tra timezone thị trường.
- H1/H7/H30 theo quan sát, ngày cuối tuần bị loại, lịch nghỉ lễ chưa xác minh. Các ngày mâu thuẫn bị cách ly, không xác định được bản ghi đúng.
- EDA toàn chuỗi chỉ mô tả; chẩn đoán dùng lịch sử trước 2018, không dùng làm feature. Không dùng các hình test để tuning.
- ADF/KPSS mức 5%; bất đồng kết luận chưa rõ. STL 252 là giả định. ACF/PACF chỉ gợi ý bậc ARIMA, không quyết định tự động.
- K=3 cố định theo mục tiêu đề tài; thử K khác chỉ để phân tích. Nhãn Bear/Sideways/Bull là thứ hạng tương đối theo return, không phải ground truth. Silhouette lấy tối đa 1000 mẫu, DB toàn train.
- Pearson >0.95 ưu tiên MI cao, giữ cả MI=0 nếu không dư thừa để tránh loại tương tác phi tuyến một cách máy móc. Top N có thể ít hơn N.
- Không tuyên bố đã loại mọi leakage hoặc cải thiện độ chính xác dự báo. Chưa thực nghiệm forecast.

##  — bổ sung minh chứng học thuật
- `academic_reports.py`: phân loại thuộc tính, tỷ lệ thiếu, histogram, tần suất event, kiểm tra miền, từ điển feature và đối chiếu target.
- `missing_details.csv`: từng ô được forward fill và ngày nguồn; giới hạn 5 là giả định, không phải tham số tối ưu.
- `outlier_details.csv`: OF=1.5, EVF=3; ngưỡng từ train, không capping. DM2 slide 43–44 định nghĩa hai mức nhưng không bắt buộc EVF=3.
- `stationarity.csv`: critical values, lag, p-value, cảnh báo; ADF autolag AIC chọn lag theo tiêu chí thông tin. ACF/PACF có CI 95%, STL có strength; period 252 là giả định.
- `feature_reduction.csv`, `correlation_matrix.csv`: số feature thực tế sau lọc; MI=0 có thể giữ để ablation. Heatmap có nhãn.
- `train_scaled.csv`, `val_scaled.csv`, `test_scaled.csv` ở holdout: chỉ scale base features; target và cluster giữ nguyên. Fold lưu scaler riêng, dùng transform khi train. Không scale lại file đã scale.
- `cluster_dimensions_experiment.csv`: cùng train thử 2/3 chiều và K=2..6; SSE, AIC/BIC chỉ so sánh trong cùng không gian đặc trưng. `cluster_centroids.csv` ở đơn vị gốc, `regime_profiles.csv` mô tả các split. K=3 vẫn là giả định nghiệp vụ, không tự chọn bằng test.
- `exclusions.csv`: warm-up, thiếu feature, thiếu target và purge. Báo cáo fold có số dòng sau purge.