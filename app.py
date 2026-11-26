import streamlit as st

st.set_page_config(
    page_title="Gold Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("GOLD INTELLIGENCE")

st.sidebar.caption(
    "Market Regime & Gold Price Forecasting"
)

navigation = st.navigation({
    "OVERVIEW": [
        st.Page(
            "pages/01_dashboard.py",
            title="Dashboard",
            default=True,
        ),
    ],
    "DATA ANALYSIS": [
        st.Page(
            "pages/02_dataset.py",
            title="Dataset",
        ),
        st.Page(
            "pages/03_eda.py",
            title="EDA",
        ),
        st.Page(
            "pages/04_preprocessing.py",
            title="Preprocessing",
        ),
        st.Page(
            "pages/05_market_regime.py",
            title="Market Regime",
        ),
    ],
    "FORECASTING": [
        st.Page(
            "pages/06_forecast_results.py",
            title="Forecast Results",
        ),
        st.Page(
            "pages/07_model_comparison.py",
            title="Model Comparison",
        ),
        st.Page(
            "pages/08_reports.py",
            title="Reports",
        ),
    ],
})

st.sidebar.divider()
st.sidebar.caption("Results Dashboard")
st.sidebar.caption("Version 1.0.0")

navigation.run()
