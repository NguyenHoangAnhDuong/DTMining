import streamlit as st

from components import charts, loaders
from components.metric_cards import kpi_row, page_header
from source.preprocessing_v3.config import EVENTS, LOGRET_COLS, MACRO, MARKET

page_header("Khám phá dữ liệu", "Diễn biến giá, tương quan và phân phối lợi suất.", "🔎")

data = loaders.base_dataset()
if data.empty:
    st.stop()

lr = data["Gold Price_logret"].dropna()
kpi_row([
    ("Lợi suất ngày TB", f"{lr.mean() * 100:.3f}%"),
    ("Biến động năm hoá", f"{lr.std() * (252 ** .5) * 100:.1f}%"),
    ("Ngày tăng mạnh nhất", f"{lr.max() * 100:+.2f}%", None, f"{data.loc[lr.idxmax(), 'Date']:%d/%m/%Y}"),
    ("Ngày giảm mạnh nhất", f"{lr.min() * 100:+.2f}%", None, f"{data.loc[lr.idxmin(), 'Date']:%d/%m/%Y}"),
])


@st.cache_data(show_spinner=False)
def _corr(cols: tuple[str, ...]):
    c = loaders.base_dataset()[list(cols)].corr()
    c.index = c.columns = [x.replace("_logret", "") for x in cols]
    return c


@st.cache_data(show_spinner=False)
def _events_per_year():
    df = loaders.base_dataset()
    ev = [c for c in df.columns if c.strip() in {e.strip() for e in EVENTS}]
    return df.groupby(df["Date"].dt.year)[ev].sum().sum(axis=1).rename("events").reset_index()


@st.fragment
def series_panel():
    with st.container(border=True):
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        cols = c1.multiselect("Biến", [c for c in MARKET + MACRO if c in data],
                              default=["Gold Price", "Silver", "S&P 500"], key="eda_cols")
        norm = c2.toggle("Chuẩn hoá (kỳ đầu = 100)", value=True, key="eda_norm")
        if cols:
            charts.show(charts.multi_line(data, cols, norm), key="eda_series")


series_panel()

left, right = st.columns([1.3, 1])
with left, st.container(border=True):
    st.markdown("##### Tương quan log-return (chuỗi dừng)")
    cols = tuple(f"{c}_logret" for c in LOGRET_COLS if f"{c}_logret" in data)
    charts.show(charts.heatmap(_corr(cols), 560), key="eda_corr")
with right:
    with st.container(border=True):
        st.markdown("##### Phân phối log-return vàng (%)")
        charts.show(charts.histogram(lr * 100, "log-return %"), key="eda_hist")
    with st.container(border=True):
        st.markdown("##### Số sự kiện toàn cầu theo năm")
        charts.show(charts.grouped_bars(_events_per_year(), "Date", {"events": ("Sự kiện", "#D4AF37")},
                                        height=200), key="eda_events")
