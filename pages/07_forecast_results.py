import streamlit as st

from components import charts, loaders
from components.filters import horizon_select, model_select, split_select
from components.metric_cards import page_header
from source.services.constants import BRANCH_LABELS, COLORS, REGIME_LABELS, REGIMES

page_header("Kết quả dự đoán", "Backtest trên walk-forward, Validation và Test: giá thực tế so với dự đoán của hai nhánh.", "Kết quả")

models = loaders.available_models()
if not models:
    st.error("Chưa có kết quả trong `results/forecasting/`.")
    st.stop()


@st.cache_data(show_spinner=False)
def _slice(model: str, horizon: int, split: str):
    p = loaders.predictions(model)
    d = p[(p["horizon"] == horizon) & (p["split"] == split)].copy()
    d["abs_err"] = (d["y_pred"] - d["y_true"]).abs()
    return d


@st.fragment
def results_panel():
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1.4, 2], vertical_alignment="bottom")
        with c1:
            model = model_select(models)
        with c2:
            horizon = horizon_select()
        with c3:
            split = split_select()

    d = _slice(model, horizon, split)
    if d.empty:
        st.info("Không có dự đoán cho cấu hình này.")
        return

    m = loaders.metrics(model)
    m = m[(m["horizon"] == horizon) & (m["split"] == split)].set_index("branch")
    if not m.empty:
        cols = st.columns(4)
        for col, metric, fmt, better_low in [
            (cols[0], "MAE", "{:,.2f}", True), (cols[1], "RMSE", "{:,.2f}", True),
            (cols[2], "MASE", "{:.4f}", True), (cols[3], "DA", "{:.1f}%", False),
        ]:
            if {"base", "kmeans"} <= set(m.index):
                kv, bv = m.loc["kmeans", metric], m.loc["base", metric]
                col.metric(f"{metric} · có phân lớp", fmt.format(kv), f"{kv - bv:+.3f} so với không phân lớp",
                           delta_color="inverse" if better_low else "normal")

    with st.container(border=True):
        st.markdown("##### Giá thực tế vs dự đoán")
        branches = st.pills("Nhánh", list(BRANCH_LABELS), default=list(BRANCH_LABELS),
                            format_func=lambda b: BRANCH_LABELS[b], selection_mode="multi", key="res_br")
        if branches:
            charts.show(charts.predictions_chart(d, branches), key="res_pred")

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("##### Lợi thế tích lũy của nhánh K-Means")
        st.caption("Σ(|sai số base| − |sai số kmeans|). Đường đi lên nghĩa là nhánh có phân lớp sai số nhỏ hơn.")
        charts.show(charts.cumulative_advantage(d), key="res_cum")
    with c2, st.container(border=True):
        st.markdown("##### MAE theo giai đoạn thị trường")
        reg = d.groupby(["regime_label", "branch"])["abs_err"].mean().unstack().reindex(REGIMES).dropna(how="all")
        reg = reg.reset_index().assign(nhãn=lambda x: x["regime_label"].map(REGIME_LABELS))
        charts.show(charts.grouped_bars(reg, "nhãn", {"base": ("Không phân lớp", COLORS["base"]),
                                                      "kmeans": ("Có phân lớp", COLORS["kmeans"])},
                                        height=300, yaxis_title="MAE (USD)"), key="res_reg")

    with st.expander("Bảng chỉ số đầy đủ (gồm naive)"):
        st.dataframe(m.reset_index().style.format(precision=4), hide_index=True)
    with st.expander("Dữ liệu dự đoán"):
        st.dataframe(d.drop(columns="abs_err"), hide_index=True, height=360)
        st.download_button("Tải CSV", loaders.to_csv(d), f"predictions_{model}_h{horizon}_{split}.csv",
                           "text/csv", on_click="ignore")


results_panel()
