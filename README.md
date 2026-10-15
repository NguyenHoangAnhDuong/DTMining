# Thông tin dự án

## "Bài toán

Phân tích giai đoạn thị trường và dự đoán giá vàng dựa trên dữ liệu giá vàng lịch sử, các chỉ số kinh tế, tài chính và sự kiện toàn cầu.

## Thuật toán và mô hình

Khai phá và tiền xử lý:
    - Data Cleaning, xử lý dữ liệu thiếu, phát hiện dữ liệu bất thường, Feature Engineering (lag, rolling mean/std, return, biến sự kiện), Feature Selection, chia dữ liệu theo chuỗi thời gian.
Phân nhóm giai đoạn thị trường:
    - K-Means / Gaussian Mixture Model để gán nhãn giai đoạn tăng – đi ngang – giảm, đánh giá bằng Silhouette và Davies-Bouldin.
Mô hình chuỗi thời gian thống kê:
    - SARIMAX (dùng thêm biến ngoại sinh USD index, dầu, lãi suất), Exponential Smoothing (Holt-Winters); chọn tham số bằng auto_arima theo AIC/BIC và kiểm tra phần dư bằng Ljung-Box.
Mô hình học máy:
    - XGBoost, LightGBM, Prophet
Mô hình cơ sở:
    - Naive Forecast và Seasonal Naive
Mô hình Deep learning:
    - Sử dụng Long Short-Term Memory (LSTM) để khai thác sự phụ thuộc theo thời gian và so sánh hiệu quả dự báo với các mô hình thống kê và Machine Learning.
Tập dữ liệu:
    - EGPBD – An Event-based Gold Price Benchmark Dataset – 8.030 bản ghi, 36 biến, 01/2001–12/2022 (giá vàng, chỉ số kinh tế/tài chính, giá hàng hóa, sự kiện toàn cầu).
Link:
    - https://data.mendeley.com/datasets/y6vj5kzfpz/1

## Quy trình tiền xử lý

1. Khám phá dữ liệu
2. Kiểm tra dữ liệu
3. Xử lý missing/duplicate/outlier
4. Xử lý dữ liệu thời gian
5. Kiểm định tính dừng (ADF, KPSS), sai phân/log-return, ACF/PACF, phân rã xu hướng – mùa vụ
6. Feature Engineering
7. Feature Selection
8. phân nhóm giai đoạn thị trường
9. Loại bỏ Data Leakagechia Train/Validation/Test theo thời gian (kèm walk-forward validation)
10. chuẩn hóa (chỉ fit trên train)

## Ứng dụng demo

Ứng dụng dự đoán giá vàng từ các mô hình đã huấn luyện.

Chức năng:
    1. Nhập dữ liệu
    2. Tiền xử lý
    3. Chọn mô hình (Naive  / SARIMAX / ML)
    4. Chọn khoảng dự đoán
    5. Dự đoán giá vàng và khoảng tin cậy
    6. Hiển thị giai đoạn thị trường hiện tại
    7. So sánh các mô hình
    8. Phân tích Feature Importance (mô hình cây)

## Đơn vị đo hiệu suất:

MAE, RMSE, MAPE, R², MASE, Directional Accuracy.
