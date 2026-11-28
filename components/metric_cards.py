"""Thẻ KPI, nhãn giai đoạn, tiêu đề trang."""
import html

import streamlit as st

from source.services.constants import REGIME_COLORS, REGIME_LABELS


def page_header(title: str, subtitle: str = "", icon: str = ""):
    st.markdown(
        f"<div class='gi-header'><h1>{icon} {html.escape(title)}</h1>"
        f"<p>{html.escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def regime_badge(regime: str | None) -> str:
    if not regime:
        return "<span class='gi-badge gi-muted'>Không phân lớp</span>"
    c = REGIME_COLORS.get(regime, "#94A3B8")
    return (f"<span class='gi-badge' style='color:{c};border-color:{c}66;background:{c}1A'>"
            f"● {REGIME_LABELS.get(regime, regime)} <small>({regime})</small></span>")


def kpi_row(items: list[tuple]):
    """items = [(label, value, delta?, help?)…] → hàng st.metric có viền."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value, *rest = item
        delta = rest[0] if len(rest) > 0 else None
        help_ = rest[1] if len(rest) > 1 else None
        col.metric(label, value, delta, help=help_, border=True)


def callout(text: str, kind: str = "info"):
    st.markdown(f"<div class='gi-callout gi-{kind}'>{text}</div>", unsafe_allow_html=True)


def fmt_usd(v, digits: int = 1) -> str:
    return "—" if v is None else f"{v:,.{digits}f} $"
