import streamlit as st

from components import charts, loaders
from components.metric_cards import callout, kpi_row, page_header, regime_badge
from source.services.constants import COLORS, MODEL_NAMES

page_header(
    "Gold Intelligence Dashboard",
    "Phân lớp giai đoạn thị trường (K-Means) trước khi dự đoán có giúp dự báo giá vàng tốt hơn không?",
    "📈",
)

data = loaders.base_dataset()
if data.empty:
    st.error("Chưa có `dataset/processed_v3/base_dataset.csv`. Chạy `python -m source.run_pipeline`.")
    st.stop()

regimes = loaders.regime_history()
last = data.iloc[-1]
prev_month = data.iloc[-22]
current_regime = regimes["market_phase"].iloc[-1] if not regimes.empty else None
cmp = loaders.forecast_summary("comparison")
ints = loaders.forecast_summary("intervals")

kpi_row([
    ("Giá vàng gần nhất", f"{last['Gold Price']:,.1f} $",
     f"{(last['Gold Price'] / prev_month['Gold Price'] - 1) * 100:+.2f}% / 1 tháng", f"Phiên {last['Date']:%d/%m/%Y}"),
    ("Số phiên giao dịch", f"{len(data):,}", None, f"{data['Date'].min():%Y} – {data['Date'].max():%Y}"),
    ("Mô hình", len(loaders.available_models()), None, ", ".join(MODEL_NAMES.get(m, m) for m in loaders.available_models())),
    ("Khoảng dự đoán", "1 · 7 · 30", None, "Số phiên giao dịch phía trước"),
])

st.markdown("")
left, right = st.columns([2.3, 1])
with left:
    with st.container(border=True):
        st.markdown("##### Giá vàng theo giai đoạn thị trường")
        charts.show(charts.price_by_regime(regimes), key="dash_price")
with right:
    with st.container(border=True):
        st.markdown("##### Giai đoạn hiện tại")
        st.markdown(regime_badge(current_regime), unsafe_allow_html=True)
        if current_regime:
            run = (regimes["market_phase"] != regimes["market_phase"].shift()).cumsum()
            days = int((run == run.iloc[-1]).sum())
            st.caption(f"Đã kéo dài {days} phiên · tính đến {regimes['Date'].iloc[-1]:%d/%m/%Y}")
        st.markdown("##### Phân bố giai đoạn (toàn bộ lịch sử)")
        charts.show(charts.regime_share(regimes), key="dash_share")

st.markdown("### Trả lời câu hỏi nghiên cứu")
if cmp.empty:
    st.info("Chưa có báo cáo so sánh. Chạy `python -m source.model.report_forecast`.")
else:
    sig = cmp["p_value"] < 0.05
    better = int((sig & (cmp["DM_stat"] < 0)).sum())
    worse = int((sig & (cmp["DM_stat"] > 0)).sum())
    n = len(cmp)
    if better > worse:
        kind, verdict = "good", "K-Means **giúp** dự đoán điểm tốt hơn ở một số khoảng dự đoán."
    elif worse > better:
        kind, verdict = "bad", "K-Means làm dự đoán điểm **kém hơn** ở một số khoảng dự đoán."
    else:
        kind, verdict = "info", "**Chưa có bằng chứng** K-Means cải thiện dự đoán điểm (không khác biệt có ý nghĩa thống kê)."
    w_txt = ""
    if not ints.empty:
        w_txt = (f" Với khoảng dự đoán, Winkler score thay đổi trung bình **{ints['Δwinkler_%'].mean():+.2f}%** "
                 f"(âm = nhánh K-Means tốt hơn), có ý nghĩa ở {int((ints['p_value'] < 0.05).sum())}/{len(ints)} cấu hình.")
    callout(
        f"Trên tập Test 2020–2022, kiểm định Diebold-Mariano: K-Means tốt hơn có ý nghĩa ở <b>{better}/{n}</b>, "
        f"kém hơn ở <b>{worse}/{n}</b> cấu hình (model × h). ΔMAE trung bình "
        f"<b>{cmp['ΔMAE_%'].mean():+.3f}%</b>.<br>" + verdict.replace("**", "") + w_txt.replace("**", ""),
        kind,
    )

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("##### MAE Test — có vs không phân lớp")
        d = cmp.assign(h="h = " + cmp["horizon"].astype(str))
        charts.show(charts.grouped_bars(d, "h", {"MAE_base": ("Không phân lớp", COLORS["base"]),
                                                  "MAE_kmeans": ("Có phân lớp", COLORS["kmeans"])},
                                        yaxis_title="USD"), key="dash_mae")
    with c2, st.container(border=True):
        st.markdown("##### MASE — < 1 là tốt hơn naive")
        fig = charts.grouped_bars(d, "h", {"MASE_base": ("Không phân lớp", COLORS["base"]),
                                           "MASE_kmeans": ("Có phân lớp", COLORS["kmeans"])})
        fig.add_hline(y=1, line=dict(color="#F87171", dash="dash"), annotation_text="naive")
        fig.update_yaxes(range=[min(0.9, d[["MASE_base", "MASE_kmeans"]].min().min() - 0.02),
                                max(1.1, d[["MASE_base", "MASE_kmeans"]].max().max() + 0.02)])
        charts.show(fig, key="dash_mase")

st.markdown("### Đi tiếp")
cols = st.columns(4)
links = [
    ("pages/06_forecast.py", "🔮 Dự đoán giá vàng", "Chọn nhánh, mô hình, h và dự đoán ngay."),
    ("pages/05_market_regime.py", "🫧 Giai đoạn thị trường", "Kết quả K-Means: chất lượng cụm, thống kê."),
    ("pages/08_model_comparison.py", "⚖️ So sánh có / không phân lớp", "DM test, khoảng dự đoán, theo giai đoạn."),
    ("pages/09_reports.py", "📄 Báo cáo", "Xem & tải báo cáo đầy đủ."),
]
for col, (page, label, desc) in zip(cols, links):
    with col, st.container(border=True):
        st.page_link(page, label=f"**{label}**")
        st.caption(desc)
