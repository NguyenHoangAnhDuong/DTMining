import pandas as pd
import streamlit as st

from components import charts, loaders
from components.metric_cards import page_header
from source.services.constants import FIGURES_DIR

page_header("Tiền xử lý", "Chỉ làm sạch và chống nhiễu — không tạo feature, không chuẩn hoá ở bước này.", "🧹")

STEPS = pd.DataFrame([
    ("01", "step01_cleaning.py", "Cách ly ngày không hợp lệ, xử lý bản ghi trùng / xung đột, chuyển kiểu số"),
    ("02", "step02_time_series.py", "Loại bỏ ngày cuối tuần (trước khi điền thiếu)"),
    ("03", "step03_missing.py", "Nội suy theo thời gian cho giá, forward-fill cho biến sự kiện"),
    ("04", "step04_outlier.py", "Phát hiện outlier — chỉ báo cáo, không cắt"),
    ("05", "step05_stationarity.py", "Kiểm định ADF/KPSS, thêm cột _logret/_diff, ACF/PACF, STL"),
    ("06", "step06_targets.py", "Tạo target đa khung h = 1, 7, 30"),
    ("07", "step07_split.py", "Chia Train 2001–2017 / Val 2018–2019 / Test 2020–2022"),
    ("09", "step09_clustering.py", "K-Means gán giai đoạn thị trường (fit trên Train)"),
], columns=["Bước", "File", "Nội dung"])

st.dataframe(STEPS, hide_index=True)

tab_st, tab_out, tab_fig = st.tabs(["Tính dừng (ADF / KPSS)", "Outlier", "Hình minh hoạ"])


def _color_conclusion(v):
    if v == "stationary":
        return "color:#4ADE80"
    if isinstance(v, str) and "non" in v:
        return "color:#F87171"
    return ""


with tab_st:
    rep = loaders.preprocessing_report("stationarity")
    if rep.empty:
        st.info("Chưa có báo cáo.")
    else:
        c1, c2 = st.columns([1, 3])
        transform = c1.radio("Dạng chuỗi", sorted(rep["transform"].unique()), key="st_tf")
        view = rep[rep["transform"] == transform]
        n_ok = int((view["conclusion"] == "stationary").sum())
        c1.metric("Chuỗi dừng", f"{n_ok}/{len(view)}", border=True)
        c2.dataframe(
            view.style.map(_color_conclusion, subset=["conclusion"])
                .format({"adf_stat": "{:.3f}", "adf_p": "{:.4f}", "kpss_stat": "{:.3f}", "kpss_p": "{:.3f}"}),
            hide_index=True, height=420,
        )
        st.caption("ADF: H0 = có nghiệm đơn vị (p < 0.05 → dừng). KPSS: H0 = dừng (p < 0.05 → không dừng).")

with tab_out:
    summ = loaders.preprocessing_report("outlier")
    if not summ.empty:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            d = summ.sort_values("n_outliers", ascending=False)
            charts.show(charts.grouped_bars(d, "feature", {"n_outliers": ("Số outlier (IQR)", "#F59E0B")},
                                            height=360), key="out_bar")
        c2.dataframe(summ.style.format(precision=2), hide_index=True, height=360)
        det = loaders.preprocessing_report("outlier_details")
        if not det.empty:
            with st.expander(f"Chi tiết {len(det):,} điểm outlier"):
                feat = st.selectbox("Biến", sorted(det["feature"].unique()))
                st.dataframe(det[det["feature"] == feat], hide_index=True)
        st.caption("Outlier chỉ được báo cáo, không cắt bỏ — biến động mạnh của thị trường là tín hiệu thật.")

with tab_fig:
    figs = {
        "Phân rã STL giá vàng": "stl_gold_price.png",
        "ACF / PACF log-return vàng": "acf_pacf_gold_logret.png",
        "Boxplot outlier": "outlier_boxplots.png",
    }
    for title, f in figs.items():
        path = FIGURES_DIR / f
        if path.is_file():
            st.markdown(f"##### {title}")
            st.image(str(path), width="stretch")
