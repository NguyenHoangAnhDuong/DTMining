# Gold Intelligence — Web app (Streamlit)

Phân tích giai đoạn thị trường và dự đoán giá vàng đa khung thời gian.
Web app trả lời câu hỏi của dự án: **phân lớp giai đoạn thị trường (K-Means) trước khi dự đoán có giúp dự báo giá vàng tốt hơn không?**

## Cài đặt & chạy

```bash
python -m pip install -r requirements.txt
python -m source.run_pipeline          # lần đầu: tạo dữ liệu, model, kết quả, báo cáo
python -m streamlit run app.py         # mở http://localhost:8501
```

App **đọc** kết quả có sẵn trong `results/` và **dự đoán trực tiếp** bằng model đã train
(`results/forecasting/<model>/models/`, `dataset/processed_v3/models/kmeans.joblib`). App không train lại model.
Sau khi chạy lại pipeline, bấm **↻ Nạp lại kết quả** ở sidebar để xoá cache.

## Các trang

| Nhóm | Trang | Nội dung |
| --- | --- | --- |
| Tổng quan | **Dashboard** | Giá gần nhất, giai đoạn hiện tại, giá theo giai đoạn, trả lời nhanh câu hỏi nghiên cứu (DM test, MAE, MASE) |
| Dự đoán | **Dự đoán giá vàng** | Luồng dự đoán chính (xem bên dưới) |
| Phân tích dữ liệu | Dữ liệu | Dữ liệu gốc / đã xử lý, nhóm biến, chia Train/Val/Test, bản ghi bị cách ly; tải CSV |
| | Khám phá dữ liệu | Diễn biến các biến (chuẩn hoá), tương quan log-return, phân phối lợi suất, sự kiện theo năm |
| | Tiền xử lý | Các bước step01–09, kiểm định ADF/KPSS, outlier, hình STL / ACF / PACF |
| | Giai đoạn thị trường | Chất lượng cụm (Silhouette, DB, CH), timeline giai đoạn, thống kê từng giai đoạn, chọn k, chọn feature |
| Kết quả | Kết quả dự đoán | Backtest thực tế vs dự đoán của 2 nhánh theo h và tập (walk-forward / Val / Test), lợi thế tích lũy, MAE theo giai đoạn |
| | So sánh có / không phân lớp | ΔMAE, Diebold-Mariano, DA; coverage & Winkler của khoảng dự đoán; kết quả theo giai đoạn; feature importance |
| | Báo cáo | Xem `result_forecast.md`, `result_k_mean.md` kèm hình; tải `.md` / `.zip` toàn bộ kết quả; tạo lại báo cáo |

## Luồng dự đoán (trang *Dự đoán giá vàng*)

1. **Chọn có phân lớp trước khi dự đoán hay không** (base / kmeans).
2. **Chọn mô hình** (các model có trong `MODEL_MODULES`).
3. **Chọn khoảng dự đoán** h = 1, 7, 30 phiên.
4. **Chọn ngày dự đoán**: mặc định là phiên cuối của dữ liệu. Chọn ngày trong quá khứ để đối chiếu với giá thực tế.
5. App gọi `predict_latest(history, branch, horizon)` với 260 phiên lịch sử tính đến ngày đã chọn:
   - *Không phân lớp:* tạo feature → model nhánh `base` → khoảng tin cậy chung.
   - *Có phân lớp:* `assign_regime` (K-Means) → tạo feature + nhãn giai đoạn → model nhánh `kmeans` → khoảng tin cậy theo giai đoạn.
6. Hiển thị giá dự đoán, khoảng 80% / 95%, giai đoạn hiện tại, nón dự đoán trên biểu đồ, và độ chính xác lịch sử (Test) của cấu hình đó.
7. Bật **So sánh cả 2 nhánh** để xem hai nhánh trên cùng biểu đồ. Bảng kết quả tải được dưới dạng CSV / JSON.

Kết quả cập nhật ngay khi đổi lựa chọn, không cần bấm nút. Lựa chọn (model, h, nhánh, ngày) được giữ khi chuyển trang.

## Hiệu năng

| Kỹ thuật | Ở đâu | Tác dụng |
| --- | --- | --- |
| `st.cache_data` | `components/loaders.py` | Mỗi CSV / JSON / MD chỉ đọc 1 lần cho mọi phiên; kết quả dự đoán cache theo (model, nhánh, h, ngày) |
| `st.cache_resource` | `loaders._warm_model`, CSS | Import module model (xgboost, sklearn…) 1 lần cho cả server |
| `@st.fragment` | Trang Dự đoán, Kết quả dự đoán, EDA | Đổi tham số chỉ chạy lại khối đó, không chạy lại cả trang |
| Plotly `Scattergl` (WebGL) | `components/charts.py` | Vẽ chuỗi ~6.000 điểm mượt, có zoom và range selector |
| Tính trước | `results/` | So sánh, DM test, kết quả theo giai đoạn đã được pipeline tính sẵn; app chỉ đọc |

Thời gian đo với AppTest: chuyển trang khoảng 0,1–0,7 giây khi đã có cache; đổi tham số dự đoán khoảng 0,1–0,4 giây (lần đầu cho mỗi cấu hình), sau đó trả ngay từ cache.

## Cấu trúc

```text
app.py                      # điều hướng (st.navigation), CSS, sidebar
pages/
  01_dashboard.py           02_dataset.py         03_eda.py
  04_preprocessing.py       05_market_regime.py   06_forecast.py
  07_forecast_results.py    08_model_comparison.py 09_reports.py
components/
  loaders.py                # MỌI truy cập dữ liệu / dự đoán đi qua đây (có cache)
  charts.py                 # biểu đồ Plotly + theme chung
  filters.py                # bộ chọn model / h / nhánh / tập (giữ giá trị khi đổi trang)
  metric_cards.py           # tiêu đề trang, thẻ KPI, nhãn giai đoạn, callout
  report_view.py            # hiển thị markdown báo cáo kèm hình
  sidebar.py                # sidebar + nạp CSS
source/services/
  constants.py              # đường dẫn (từ config.py), model, nhãn, màu
  data_loader.py            # đọc dữ liệu
  result_loader.py          # đọc kết quả / báo cáo
  forecast_service.py       # dự đoán tại 1 ngày (không phụ thuộc Streamlit)
assets/styles/custom.css    # giao diện tối + vàng
.streamlit/config.toml      # theme, cổng
```

## Thêm model mới vào app

1. Train model theo hướng dẫn trong `README.md` (ghi kết quả vào `results/forecasting/<model>/` theo `docs/result_contract.md`).
2. Thêm vào `source/services/constants.py`:
   - `MODELS` và `MODEL_NAMES` → model xuất hiện ở các trang kết quả / so sánh.
   - `MODEL_MODULES` (module có `predict_latest(history, branch, horizon)`) → model xuất hiện ở trang Dự đoán.
3. Chạy lại `python -m source.model.report_forecast`, rồi bấm **↻ Nạp lại kết quả**.
