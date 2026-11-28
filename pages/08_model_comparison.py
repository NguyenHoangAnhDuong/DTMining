import streamlit as st

from components import charts, loaders
from components.filters import horizon_select, model_select
from components.metric_cards import callout, page_header
from source.services.constants import COLORS, REGIME_LABELS

page_header("So sánh có / không phân lớp",
            "Cùng model, cùng feature gốc, cùng target. Khác biệt duy nhất là thông tin giai đoạn K-Means.", "Kết quả")

cmp = loaders.forecast_summary("comparison")
ints = loaders.forecast_summary("intervals")
by_reg = loaders.forecast_summary("by_regime")
int_reg = loaders.forecast_summary("intervals_by_regime")
models = loaders.available_models()

if cmp.empty or not models:
    st.info("Chưa có báo cáo so sánh. Chạy `python -m source.model.report_forecast`.")
    st.stop()


def _sig_style(v):
    if not isinstance(v, str):
        return ""
    if "tốt hơn" in v.lower():
        return f"color:{COLORS['good']};font-weight:600"
    if "kém" in v.lower():
        return f"color:{COLORS['bad']};font-weight:600"
    return f"color:{COLORS['muted']}"


def _p_style(v):
    return f"color:{COLORS['good']};font-weight:600" if isinstance(v, float) and v < 0.05 else ""


model = model_select(models, key="cmp_model")
c = cmp[cmp["model"] == model]

tab_point, tab_int, tab_reg, tab_fi = st.tabs(
    ["Dự đoán điểm", "Khoảng dự đoán", "Theo giai đoạn", "Feature importance"])

with tab_point:
    callout("Kiểm định <b>Diebold-Mariano</b> trên sai số tuyệt đối (Test 2020–2022): DM &lt; 0 nghĩa là nhánh "
            "K-Means sai số nhỏ hơn; khác biệt <b>có ý nghĩa</b> khi p &lt; 0.05.")
    l, r = st.columns(2)
    with l, st.container(border=True):
        st.markdown("##### ΔMAE (%), âm là K-Means tốt hơn")
        charts.show(charts.delta_bars(c.assign(h="h = " + c["horizon"].astype(str)), "h", "ΔMAE_%"), key="cmp_dmae")
    with r, st.container(border=True):
        st.markdown("##### Directional Accuracy (%)")
        charts.show(charts.grouped_bars(c.assign(h="h = " + c["horizon"].astype(str)), "h",
                                        {"DA_base": ("Không phân lớp", COLORS["base"]),
                                         "DA_kmeans": ("Có phân lớp", COLORS["kmeans"])}, height=300),
                    key="cmp_da")
    st.dataframe(c.style.map(_sig_style, subset=["kết_luận"]).map(_p_style, subset=["p_value"])
                 .format(precision=4), hide_index=True)

with tab_int:
    callout("<b>Coverage</b> càng sát mức danh nghĩa (80% / 95%) càng tốt; <b>Winkler</b> = độ rộng + phạt "
            "khi giá thực nằm ngoài khoảng, càng nhỏ càng tốt. Nhánh K-Means dùng độ rộng riêng cho từng giai đoạn.")
    i = ints[ints["model"] == model].copy()
    i["cấu hình"] = "h=" + i["horizon"].astype(str) + " · " + i["level_%"].astype(str) + "%"
    l, r = st.columns(2)
    with l, st.container(border=True):
        st.markdown("##### ΔWinkler (%), âm là K-Means tốt hơn")
        charts.show(charts.delta_bars(i, "cấu hình", "Δwinkler_%"), key="cmp_dw")
    with r, st.container(border=True):
        st.markdown("##### Coverage thực tế vs danh nghĩa")
        fig = charts.grouped_bars(i, "cấu hình", {"coverage_%_base": ("Không phân lớp", COLORS["base"]),
                                                  "coverage_%_kmeans": ("Có phân lớp", COLORS["kmeans"])},
                                  height=300, yaxis_title="%")
        for lv in (80, 95):
            fig.add_hline(y=lv, line=dict(color=COLORS["rule"], dash="dot", width=1), annotation_text=f"{lv}%",
                          annotation_font=dict(size=11, color=COLORS["muted"]))
        fig.update_yaxes(range=[60, 100])
        charts.show(fig, key="cmp_cov")
    st.dataframe(i.drop(columns="cấu hình").style.map(_sig_style, subset=["kết_luận"])
                 .map(_p_style, subset=["p_value"]).format(precision=3), hide_index=True)

with tab_reg:
    horizon = horizon_select(key="cmp_h")
    br = by_reg[(by_reg["model"] == model) & (by_reg["horizon"] == horizon)]
    l, r = st.columns(2)
    with l, st.container(border=True):
        st.markdown("##### ΔMAE (%) theo giai đoạn")
        charts.show(charts.delta_bars(br, "nhãn", "ΔMAE_%"), key="cmp_reg_dmae")
        st.dataframe(br.style.format(precision=3), hide_index=True)
    with r, st.container(border=True):
        lvl = st.segmented_control("Mức tin cậy", [80, 95], default=80, key="cmp_lvl", required=True,
                                   format_func=lambda x: f"{x}%")
        ir = int_reg[(int_reg["model"] == model) & (int_reg["horizon"] == horizon) & (int_reg["level_%"] == lvl)]
        ir = ir.assign(nhãn=ir["regime"].map(REGIME_LABELS))
        st.markdown(f"##### Coverage {lvl}% theo giai đoạn")
        fig = charts.grouped_bars(ir, "nhãn", {"coverage_%_base": ("Không phân lớp", COLORS["base"]),
                                               "coverage_%_kmeans": ("Có phân lớp", COLORS["kmeans"])},
                                  height=280, yaxis_title="%")
        fig.add_hline(y=lvl, line=dict(color=COLORS["rule"], dash="dot", width=1))
        charts.show(fig, key="cmp_reg_cov")
        st.dataframe(ir.drop(columns="nhãn").style.format(precision=2), hide_index=True)

with tab_fi:
    fi = loaders.feature_importance(model)
    if fi.empty:
        st.info("Model không có feature importance.")
    else:
        c1, c2, c3 = st.columns([1.3, 1.5, 1])
        with c1:
            branch = st.segmented_control("Nhánh", ["base", "kmeans"], default="kmeans", key="fi_br", required=True,
                                          format_func=lambda b: "Không phân lớp" if b == "base" else "Có phân lớp")
        with c2:
            h = horizon_select(key="fi_h")
        top = c3.slider("Top", 10, 40, 20, key="fi_top")
        d = fi[(fi["branch"] == branch) & (fi["horizon"] == h)]
        st.caption("Cột màu cam là feature giai đoạn K-Means, cột xanh là feature gốc.")
        charts.show(charts.feature_importance_chart(d, top), key="cmp_fi")
        reg = d[d["feature"].str.contains("kmeans_|regime_id")]
        if not reg.empty:
            share = reg["importance_gain"].sum() / d["importance_gain"].sum() * 100
            rank = d["importance_gain"].rank(ascending=False)[reg.index].min()
            st.markdown(f"Feature giai đoạn chiếm **{share:.1f}%** tổng gain; cao nhất xếp hạng **#{int(rank)}** "
                        f"/ {len(d)} feature.")
