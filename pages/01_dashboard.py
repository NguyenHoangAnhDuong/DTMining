import streamlit as st

from src.constants import (
    MODELS,
    MODEL_NAMES,
    TASKS,
    HORIZONS,
    CLUSTERING_ALGORITHMS,
)

from src.result_loader import load_all_metrics

st.title("Gold Intelligence Dashboard")

st.caption(
    "Market Regime Analysis & "
    "Multi-Horizon Gold Price Forecasting"
)

cols = st.columns(4)

cols[0].metric(
    "Forecast Models",
    len(MODELS),
)

cols[1].metric(
    "Experiment Tasks",
    len(TASKS),
)

cols[2].metric(
    "Forecast Horizons",
    len(HORIZONS),
)

cols[3].metric(
    "Clustering Algorithms",
    len(CLUSTERING_ALGORITHMS),
)

st.divider()

st.subheader("Forecast Models")

cols = st.columns(5)

for col, model in zip(cols, MODELS):
    col.info(MODEL_NAMES[model])

st.subheader("Experiment Tasks")

left, right = st.columns(2)

with left:
    with st.container(border=True):
        st.markdown("### Task 1")
        st.write("With Clustering")
        st.caption(
            "Base Features + Cluster Features"
        )

with right:
    with st.container(border=True):
        st.markdown("### Task 2")
        st.write("Without Clustering")
        st.caption("Base Features Only")

st.subheader("Forecast Horizons")

cols = st.columns(3)

for col, horizon in zip(cols, HORIZONS):
    col.metric(
        "Horizon",
        f"h = {horizon}",
    )

st.divider()

st.subheader("Experiment Results")

metrics = load_all_metrics()

if metrics.empty:
    st.info(
        "No experiment results available. "
        "Please import results from Google Colab."
    )
else:
    st.dataframe(
        metrics,
        use_container_width=True,
    )
