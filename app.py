import streamlit as st

from components.sidebar import inject_css, render_sidebar, render_sidebar_footer

st.set_page_config(
    page_title="Gold Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
render_sidebar()

navigation = st.navigation({
    "TỔNG QUAN": [
        st.Page("pages/01_dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
    ],
    "DỰ ĐOÁN": [
        st.Page("pages/06_forecast.py", title="Dự đoán giá vàng", icon=":material/insights:"),
    ],
    "PHÂN TÍCH DỮ LIỆU": [
        st.Page("pages/02_dataset.py", title="Dữ liệu", icon=":material/table_chart:"),
        st.Page("pages/03_eda.py", title="Khám phá dữ liệu", icon=":material/query_stats:"),
        st.Page("pages/04_preprocessing.py", title="Tiền xử lý", icon=":material/cleaning_services:"),
        st.Page("pages/05_market_regime.py", title="Giai đoạn thị trường", icon=":material/bubble_chart:"),
    ],
    "KẾT QUẢ": [
        st.Page("pages/07_forecast_results.py", title="Kết quả dự đoán", icon=":material/show_chart:"),
        st.Page("pages/08_model_comparison.py", title="So sánh có / không phân lớp", icon=":material/compare_arrows:"),
        st.Page("pages/09_reports.py", title="Báo cáo", icon=":material/description:"),
    ],
})

render_sidebar_footer()

navigation.run()
