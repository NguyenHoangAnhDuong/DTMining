import streamlit as st

from components import charts, loaders
from components.metric_cards import callout, kpi_row, page_header, regime_badge
from source.services.constants import CLUSTERING_RESULTS_DIR, REGIME_COLORS, REGIME_LABELS

page_header("Giai đoạn thị trường",
            "K-Means (k = 4) trên xu hướng 60 phiên và biến động 20 phiên — fit chỉ trên Train.", "🫧")

regimes = loaders.regime_history()
km = loaders.kmeans_summary("metrics")

if not km.empty:
    kpi_row([
        (f"Silhouette · {r['split']}", f"{r['silhouette']:.3f}", None,
         f"Davies-Bouldin {r['davies_bouldin']:.3f} · Calinski-Harabasz {r['calinski_harabasz']:,.0f}")
        for _, r in km.iterrows()
    ])

callout(
    "Bốn giai đoạn được đặt tên theo tâm cụm: cụm có biến động cao nhất là <b>Biến động mạnh</b>, "
    "ba cụm còn lại xếp theo xu hướng tăng dần thành <b>Giảm</b> → <b>Tăng</b> → <b>Tăng mạnh</b>. "
    "Feature chỉ dùng quá khứ (shift 1) nên không rò rỉ thông tin.",
)

with st.container(border=True):
    c1, c2 = st.columns([3, 1])
    c1.markdown("##### Diễn biến giá vàng tô màu theo giai đoạn")
    if not regimes.empty:
        c2.markdown("Hiện tại: " + regime_badge(regimes["market_phase"].iloc[-1]), unsafe_allow_html=True)
    charts.show(charts.price_by_regime(regimes, 460), key="reg_price")

tab_stats, tab_k, tab_feat, tab_fig = st.tabs(["Thống kê từng giai đoạn", "Chọn k", "Chọn feature", "Hình minh hoạ"])

with tab_stats:
    stats = loaders.kmeans_summary("regimes")
    if not stats.empty:
        split = st.segmented_control("Tập", stats["split"].unique().tolist(), default="train",
                                     key="reg_split", required=True)
        view = stats[stats["split"] == split].drop(columns="split")
        view.insert(1, "Nhãn", view["regime"].map(REGIME_LABELS))
        st.dataframe(
            view.style.map(lambda v: f"color:{REGIME_COLORS.get(v, '')}", subset=["regime"]).format(precision=3),
            hide_index=True,
        )
    oos = loaders.kmeans_summary("oos_usefulness")
    if not oos.empty:
        st.markdown("##### Nhãn giai đoạn có thông tin về tương lai không? (ngoài mẫu)")
        st.dataframe(oos.style.format(precision=3), hide_index=True)
        st.caption("fwd_vol20_ann_pct: biến động 20 phiên tới (năm hoá). Khác biệt giữa các giai đoạn "
                   "là cơ sở để hiệu chỉnh khoảng dự đoán theo giai đoạn.")

with tab_k:
    scan = loaders.kmeans_summary("k_scan")
    if not scan.empty:
        charts.show(charts.k_scan(scan), key="reg_kscan")
        st.dataframe(scan.style.format(precision=3), hide_index=True)

with tab_feat:
    fc = loaders.kmeans_summary("feature_comparison")
    if not fc.empty:
        st.dataframe(fc.style.format(precision=3), hide_index=True)
        st.caption("avg_run_days: độ dài trung bình một đợt giai đoạn — càng dài, nhãn càng ổn định.")

with tab_fig:
    figs = {
        "Không gian phân cụm": "kmeans_cluster_scatter.png",
        "Ma trận chuyển giai đoạn": "kmeans_transition_matrix.png",
        "Lợi suất tương lai theo giai đoạn": "kmeans_forward_returns.png",
        "Biến động ngoài mẫu": "kmeans_oos_volatility.png",
    }
    cols = st.columns(2)
    for i, (title, f) in enumerate(figs.items()):
        path = CLUSTERING_RESULTS_DIR / "figures" / f
        if path.is_file():
            cols[i % 2].image(str(path), caption=title, width="stretch")
