"""
Dự đoán giá vàng — đúng luồng trong README:
  1. Chọn có phân lớp hay không  2. Chọn mô hình  3. Chọn h  4. Dự đoán
  5. Hiển thị giá, khoảng 80%/95%, giai đoạn  6. So sánh 2 nhánh của cùng model
Toàn bộ khối tương tác nằm trong @st.fragment → đổi tham số chỉ chạy lại khối này.
"""
import json

import pandas as pd
import streamlit as st

from components import charts, loaders
from components.filters import bind, branch_select, horizon_select, model_select
from components.metric_cards import callout, fmt_usd, page_header, regime_badge
from source.services.constants import BRANCH_LABELS, MODEL_MODULES, MODEL_NAMES, REGIME_LABELS
from source.services.forecast_service import min_anchor_index, result_to_frame

page_header("Dự đoán giá vàng", "Chọn cấu hình → mô hình dự đoán ngay trên dữ liệu đến ngày bạn chọn.", "🔮")

data = loaders.base_dataset()
models = [m for m in loaders.available_models() if m in MODEL_MODULES]
if data.empty or not models:
    st.error("Chưa có dữ liệu hoặc mô hình. Chạy `python -m source.run_pipeline` trước.")
    st.stop()

dates = data["Date"]
min_date, max_date = dates.iloc[min_anchor_index()].date(), dates.iloc[-1].date()


def _steps(branch: str, model: str):
    name = MODEL_NAMES.get(model, model)
    if branch == "kmeans":
        steps = ["Lịch sử ≤ ngày chọn", "K-Means gán giai đoạn", "Tạo feature + nhãn giai đoạn",
                 f"{name} · nhánh kmeans", "Khoảng tin cậy theo giai đoạn"]
    else:
        steps = ["Lịch sử ≤ ngày chọn", "Tạo feature", f"{name} · nhánh base", "Khoảng tin cậy chung"]
    st.caption("  →  ".join(f"**{i}.** {s}" for i, s in enumerate(steps, 1)))


def _result_card(r: dict):
    pct = (r["pred_price"] / r["price_t"] - 1) * 100
    arrow, color = ("▲", "#4ADE80") if pct >= 0 else ("▼", "#F87171")
    with st.container(border=True):
        st.markdown(
            f"<div class='gi-forecast-sub'>{BRANCH_LABELS[r['branch']]} · dự đoán cho "
            f"<b>{r['target_date']:%d/%m/%Y}</b> (h = {r['horizon']})</div>"
            f"<div class='gi-forecast'>{r['pred_price']:,.1f} $</div>"
            f"<div style='color:{color};font-weight:600'>{arrow} {pct:+.2f}% so với "
            f"{r['price_t']:,.1f} $ ngày {r['date']:%d/%m/%Y}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        for lv, (lo, hi) in sorted(r["intervals"].items(), key=lambda x: -float(x[0])):
            st.markdown(
                f"<div class='gi-interval'>Khoảng {int(float(lv) * 100)}%: "
                f"<b>{lo:,.1f}</b> – <b>{hi:,.1f}</b> $ "
                f"<span class='gi-forecast-sub'>(± {(hi - lo) / 2:,.0f})</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("")
        st.markdown("Giai đoạn thị trường: " + regime_badge(r["regime"]), unsafe_allow_html=True)

        if r.get("actual") is not None:
            st.divider()
            err = r["error"]
            inside = " · ".join(
                f"{int(float(lv) * 100)}% {'✅' if ok else '❌'}" for lv, ok in sorted(r["inside"].items())
            )
            st.markdown(
                f"Giá thực tế: **{r['actual']:,.1f} $** · sai số **{err:+,.1f} $** "
                f"({err / r['actual'] * 100:+.2f}%)  \nNằm trong khoảng: {inside}"
            )


def _history_quality(model: str, branch: str, horizon: int):
    m = loaders.metrics(model)
    m = m[(m["horizon"] == horizon) & (m["split"] == "test")]
    own, naive = m[m["branch"] == branch], m[m["branch"] == "naive"]
    if own.empty:
        return
    own = own.iloc[0]
    cols = st.columns(4)
    cols[0].metric("MAE Test", fmt_usd(own["MAE"]),
                   f"{own['MAE'] - naive['MAE'].iloc[0]:+.2f} so với naive" if not naive.empty else None,
                   delta_color="inverse", border=True)
    cols[1].metric("MAPE Test", f"{own['MAPE']:.2f}%", border=True)
    cols[2].metric("MASE", f"{own['MASE']:.3f}", "tốt hơn naive" if own["MASE"] < 1 else "kém naive",
                   delta_color="normal" if own["MASE"] < 1 else "inverse", delta_arrow="off", border=True,
                   help="< 1: tốt hơn dự đoán ngây thơ (giá t+h = giá t)")
    cols[3].metric("Đúng hướng (DA)", f"{own['DA']:.1f}%", border=True)


@st.fragment
def forecast_panel():
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.5, 1, 1.4, 1.1], vertical_alignment="bottom")
        with c1:
            branch = branch_select()
        with c2:
            model = model_select(models)
        with c3:
            horizon = horizon_select()
        with c4:
            akey, asave = bind("anchor_date", max_date)
            anchor = st.date_input("Dự đoán từ ngày", min_value=min_date, max_value=max_date,
                                   key=akey, format="DD/MM/YYYY", on_change=asave,
                                   help="Mặc định là phiên cuối của dữ liệu. Chọn ngày trong quá khứ để đối chiếu với giá thực tế.")
        o1, o2 = st.columns([1, 1])
        compare = o1.toggle("So sánh cả 2 nhánh (có / không phân lớp)", value=True, key="compare_both")
        lookback = o2.segmented_control("Hiển thị lịch sử", [60, 120, 250], default=120,
                                        format_func=lambda n: f"{n} phiên", key="lookback") or 120
        _steps(branch, model)

    idx = loaders.anchor_index(anchor)
    branches = [branch] + ([b for b in BRANCH_LABELS if b != branch] if compare else [])
    try:
        with st.spinner("Đang dự đoán…"):
            results = [loaders.forecast(model, b, horizon, idx) for b in branches]
    except Exception as exc:  # hiển thị lỗi thay vì làm sập trang
        st.error(f"Không dự đoán được: {exc}")
        return

    main = results[0]
    if pd.Timestamp(anchor) != main["date"]:
        st.caption(f"Ngày {anchor:%d/%m/%Y} không có phiên giao dịch → dùng phiên gần nhất {main['date']:%d/%m/%Y}.")

    left, right = st.columns([1, 2.1])
    with left:
        _result_card(main)
    with right:
        charts.show(charts.forecast_fan(data, results, lookback), key="fan")

    if compare and len(results) == 2:
        b, k = sorted(results, key=lambda r: r["branch"])
        diff = k["pred_price"] - b["pred_price"]
        w80 = lambda r: r["intervals"]["0.8"][1] - r["intervals"]["0.8"][0]
        msg = (f"Nhánh <b>có phân lớp</b> dự đoán {k['pred_price']:,.1f} $, nhánh <b>không phân lớp</b> "
               f"{b['pred_price']:,.1f} $ (chênh {diff:+,.1f} $). Độ rộng khoảng 80%: "
               f"{w80(k):,.0f} $ vs {w80(b):,.0f} $")
        if k["regime"]:
            msg += f" — khoảng của nhánh K-Means được hiệu chỉnh riêng cho giai đoạn <b>{REGIME_LABELS[k['regime']]}</b>."
        if k.get("actual") is not None:
            better = "có phân lớp" if abs(k["error"]) < abs(b["error"]) else "không phân lớp"
            msg += f"<br>Đối chiếu thực tế: nhánh <b>{better}</b> sát hơn ở lần dự đoán này."
        callout(msg, "info")

    st.markdown(f"##### Độ chính xác lịch sử của cấu hình này · {MODEL_NAMES.get(model, model)} · "
                f"{BRANCH_LABELS[branch]} · h = {horizon}")
    _history_quality(model, branch, horizon)

    table = result_to_frame(results)
    with st.expander("Bảng kết quả & tải về", expanded=False):
        st.dataframe(table, hide_index=True)
        d1, d2 = st.columns(2)
        stem = f"forecast_{model}_h{horizon}_{main['date']:%Y%m%d}"
        d1.download_button("⬇ Tải CSV", loaders.to_csv(table), f"{stem}.csv", "text/csv",
                           width="stretch", on_click="ignore")
        d2.download_button("⬇ Tải JSON", json.dumps(json.loads(table.to_json(orient="records", date_format="iso")),
                                                   ensure_ascii=False, indent=2),
                           f"{stem}.json", "application/json", width="stretch", on_click="ignore")


forecast_panel()

with st.expander("Cách đọc kết quả"):
    st.markdown(
        "- **Giá dự đoán** = giá hôm nay × exp(log-return dự đoán h phiên tới).\n"
        "- **Khoảng 80% / 95%** (conformal): hiệu chỉnh trên sai số out-of-fold 2004–2017. "
        "Nhánh *không phân lớp* dùng một độ rộng chung; nhánh *có phân lớp* dùng độ rộng theo giai đoạn.\n"
        "- **Giai đoạn thị trường**: K-Means (k = 4, fit trên Train) trên xu hướng 60 phiên và biến động 20 phiên — "
        "chỉ dùng dữ liệu quá khứ.\n"
        "- **h** tính theo phiên giao dịch (không tính cuối tuần)."
    )
