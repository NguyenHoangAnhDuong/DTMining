import streamlit as st

from components.sidebar import inject_css, render_sidebar, render_sidebar_footer

st.set_page_config(
    page_title="Gold Intelligence",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
render_sidebar()

navigation = st.navigation({
    "": [
        st.Page("pages/01_dashboard.py", title="Tổng quan", default=True),
        st.Page("pages/06_forecast.py", title="Dự đoán giá vàng"),
    ],
    "Dữ liệu & phân cụm": [
        st.Page("pages/02_dataset.py", title="Dữ liệu"),
        st.Page("pages/03_eda.py", title="Khám phá dữ liệu"),
        st.Page("pages/04_preprocessing.py", title="Tiền xử lý"),
        st.Page("pages/05_market_regime.py", title="Giai đoạn thị trường"),
    ],
    "Kết quả": [
        st.Page("pages/07_forecast_results.py", title="Kết quả dự đoán"),
        st.Page("pages/08_model_comparison.py", title="So sánh có / không phân lớp"),
        st.Page("pages/09_reports.py", title="Báo cáo"),
    ],
})

render_sidebar_footer()

navigation.run()
