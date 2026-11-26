# Giải thích kỹ thuật để báo cáo

1. **Data understanding:** xác định kiểu dữ liệu, miền giá trị, dữ liệu thiếu và phạm vi thời gian. Thống kê toàn bộ chỉ dùng mô tả, không học tham số dự báo.
2. **Cleaning:** ngày không tồn tại không được tự sửa. Trùng hoàn toàn giữ một dòng; ngày mâu thuẫn cách ly tất cả dòng để tránh chọn giá tùy ý. Ba target nguồn được lưu đối chiếu, không dùng làm nhãn.
3. **Missing:** forward fill x_t = x_{t-1} gần nhất, giới hạn 5 quan sát. Không dùng backward fill/interpolation vì có thể đưa tương lai vào quá khứ. Gold Price thiếu không điền; mẫu không đủ điều kiện bị loại sau khi tạo target.
4. **Outlier:** IQR = Q3-Q1. Ngoại lai ngoài [Q1-1.5IQR,Q3+1.5IQR]. Ngưỡng chỉ tính train; giữ cực trị thật để bảo toàn khủng hoảng. Giá dầu có thể âm nên dùng sai phân.
5. **Time series:** sắp xếp tăng dần, kiểm tra gap. H là số quan sát còn lại, không phải số ngày lịch hoặc lịch giao dịch đã xác minh.
6. **Stationarity:** ADF H0 có unit root, KPSS H0 dừng quanh mức trung bình. ADF p<.05 và KPSS p>=.05 là bằng chứng dừng; ngược lại là bằng chứng không dừng; các trường hợp khác chưa rõ. Log-return ln(P_t/P_{t-1}); difference x_t-x_{t-1}. Không bắt buộc mọi feature của cây phải dừng.
7. **ACF/PACF/STL:** ACF đo tương quan theo độ trễ; PACF loại ảnh hưởng tuyến tính của lag trung gian. STL phân rã Y=T+S+R; chu kỳ 252 giả định. Không dùng STL hai phía làm feature.
8. **Feature engineering:** lag k lấy x_{t-k}; rolling mean/std dùng cửa sổ kết thúc t-1; ratio biểu diễn quan hệ giá; calendar mô tả thời điểm. Macro/event lag1 là proxy chưa xác minh. Cluster rolling 20 gồm t theo giả định sau phiên.
9. **Target:** y=P_{t+h}; return=ln(P_{t+h}/P_t); chuyển dự báo return thành giá P_t*exp(r_hat). Tạo target trước khi bỏ warmup để giữ đúng alignment.
10. **Split:** train trước 2018, val 2018–2019, test từ 2020. Purge mẫu nếu ngày target vượt ranh giới. Expanding window chỉ dùng lịch sử train, fit lại từng fold.
11. **Selection:** MI đo phụ thuộc giữa X và y; Pearson đo tương quan tuyến tính. Xếp MI giảm dần rồi loại biến hằng và biến có |r|>.95 với biến đã giữ. Không dùng test để chọn feature.
12. **Normalization:** z=(x-mu_train)/sigma_train. Không fit val/test; cây thường không cần scale.
13. **Clustering:** K-Means tối thiểu hóa tổng bình phương khoảng cách đến centroid; GMM mô hình hóa hỗn hợp Gaussian và xác suất thành viên. Silhouette cao hơn, Davies-Bouldin thấp hơn thường tốt hơn. Nhãn theo return trung bình tương đối. Thử 2/3 feature bằng tham số CLI; thử K=2..6 trên train, không tự chọn K theo test.
14. **Validation:** kiểm tra ngày, nhãn, NaN/Inf, ranh giới và tổng xác suất. Đây không phải chứng minh mọi nguồn dữ liệu đều có sẵn đúng thời điểm.

## Minh chứng V3.2 và cách giải thích
1. EDA: phân biệt ngày (thời gian), event (số đếm rời rạc không âm; có giá trị lớn hơn 1), giá/chỉ số (định lượng). Histogram và skewness mô tả phân bố, không dùng thống kê test để chọn mô hình.
2. IQR: Q1 và Q3 là phân vị 25% và 75%. OF=1.5, EVF=3 là thiết lập thực nghiệm. Ngoại lai không đồng nghĩa lỗi; giữ biến động thật. Dùng báo cáo chi tiết để xác minh từng ngày.
3. Z-score: z=(x-mu_train)/sigma_train, sigma dùng ddof=0. Train sau transform có mean gần 0 và std gần 1 (cột hằng có std 0); val/test không bắt buộc như vậy.
4. K-means: khởi tạo K-means++, gán theo khoảng cách Euclid, cập nhật centroid bằng trung bình, lặp đến hội tụ. SSE=sum_i ||x_i-c_label(i)||². Elbow là gợi ý, không phải bằng chứng tối ưu duy nhất. GMM dùng EM và xác suất hậu nghiệm; AIC=2p-2logL, BIC=p log(n)-2logL. So sánh AIC/BIC trong cùng số chiều.
5. ADF H0 có unit root, KPSS H0 dừng quanh mức trung bình. Không bác bỏ không đồng nghĩa chứng minh H0. CI ACF/PACF 95% là xấp xỉ, không tự chọn p,q chỉ bằng biểu đồ. STL strength=max(0,1-var(R)/var(component+R)).
6. Feature dictionary ghi công thức và ví dụ đầu tiên có giá trị. Target comparison chỉ đối chiếu, không kết luận nhãn gốc sai vì định nghĩa horizon có thể khác.
7. Bear/Sideways/Bull là tên tương đối theo mean rolling return. Đọc cả volatility và tỷ lệ return dương; không coi nhãn là ground truth.
