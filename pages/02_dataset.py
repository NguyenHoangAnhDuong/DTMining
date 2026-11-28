import streamlit as st

from components import loaders
from components.metric_cards import kpi_row, page_header
from source.preprocessing_v3.config import EVENTS, MACRO, MARKET

page_header("Dữ liệu", "EGPBD — Event-based Gold Price Benchmark Dataset (01/2001 – 12/2022).", "🗂️")

base = loaders.base_dataset()
raw = loaders.raw_dataset()
if base.empty:
    st.error("Chưa có dữ liệu đã tiền xử lý.")
    st.stop()

kpi_row([
    ("Bản ghi gốc", f"{len(raw):,}" if not raw.empty else "—", None, "dataset/data_root/dataset_root.csv"),
    ("Sau tiền xử lý", f"{len(base):,}", f"{len(base) - len(raw):,} bản ghi" if not raw.empty else None,
     "Bỏ ngày không hợp lệ / trùng / cuối tuần"),
    ("Số cột", base.shape[1], None, "Gốc + _logret/_diff + target"),
    ("Giai đoạn", f"{base['Date'].min():%Y} – {base['Date'].max():%Y}"),
])

EVENT_SET = {e.strip() for e in EVENTS}
groups = {
    "Thị trường": [c for c in MARKET if c in base],
    "Vĩ mô": [c for c in MACRO if c in base],
    "Sự kiện (0/1)": [c for c in base.columns if c.strip() in EVENT_SET],
    "Biến đổi dừng": [c for c in base.columns if c.endswith(("_logret", "_diff"))],
    "Target": [c for c in base.columns if c.startswith("target_")],
}

tab_base, tab_split, tab_raw, tab_quality = st.tabs(
    ["Dữ liệu đã xử lý", "Chia Train / Val / Test", "Dữ liệu gốc", "Chất lượng dữ liệu"])

with tab_base:
    c1, c2 = st.columns([2, 1])
    picked = c1.multiselect("Nhóm cột", list(groups), default=["Thị trường", "Target"])
    y_min, y_max = int(base["Date"].dt.year.min()), int(base["Date"].dt.year.max())
    rng = c2.slider("Năm", y_min, y_max, (y_max - 2, y_max))
    cols = ["Date"] + [c for g in picked for c in groups[g]]
    view = base.loc[base["Date"].dt.year.between(*rng), cols]
    st.dataframe(view, hide_index=True, height=420,
                 column_config={"Date": st.column_config.DateColumn("Ngày", format="DD/MM/YYYY")})
    st.caption(f"{len(view):,} dòng × {len(cols)} cột")
    st.download_button("⬇ Tải base_dataset.csv", loaders.to_csv(base), "base_dataset.csv", "text/csv",
                       on_click="ignore")

    with st.expander("Mô tả các nhóm biến"):
        for g, cs in groups.items():
            st.markdown(f"**{g}** ({len(cs)}): " + ", ".join(f"`{c}`" for c in cs))

with tab_split:
    st.markdown("Chia **theo thời gian**, cùng mốc cho cả 2 nhánh (base / kmeans) — không xáo trộn để tránh rò rỉ.")
    st.dataframe(loaders.split_sizes(), hide_index=True)
    st.markdown("Nhánh **kmeans** thêm các cột `kmeans_Bear`, `kmeans_Bull`, `kmeans_StrongBull`, "
                "`kmeans_Volatile`, `regime_id`, `market_phase`.")

with tab_raw:
    if raw.empty:
        st.info("Không tìm thấy dữ liệu gốc.")
    else:
        st.dataframe(raw.head(500), hide_index=True, height=420)
        st.caption(f"Hiển thị 500 / {len(raw):,} dòng đầu")

with tab_quality:
    inv = loaders.preprocessing_report("quarantine_invalid_dates")
    conf = loaders.preprocessing_report("quarantine_conflicts")
    q1, q2 = st.columns(2)
    q1.metric("Ngày không hợp lệ (cách ly)", len(inv), border=True)
    q2.metric("Bản ghi trùng ngày / xung đột", len(conf), border=True)
    with st.expander("Bản ghi ngày không hợp lệ"):
        st.dataframe(inv, hide_index=True)
    with st.expander("Bản ghi xung đột"):
        st.dataframe(conf, hide_index=True)
    missing = base[groups["Thị trường"] + groups["Vĩ mô"]].isna().sum()
    st.markdown("**Giá trị thiếu sau tiền xử lý:** " + (
        "không còn ✅" if missing.sum() == 0
        else ", ".join(f"{k}: {v}" for k, v in missing[missing > 0].items())))
