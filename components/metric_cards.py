"""Thẻ KPI, nhãn giai đoạn, tiêu đề trang."""
import html

import streamlit as st

from source.services.constants import REGIME_COLORS, REGIME_LABELS


def page_header(title: str, subtitle: str = "", kicker: str = ""):
    """kicker: dòng nhỏ phía trên tiêu đề (tên mục), giống tiêu đề bài trên trang dữ liệu."""
    top = f"<div class='gi-kicker'>{html.escape(kicker)}</div>" if kicker else ""
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(f"<div class='gi-header'>{top}<h1>{html.escape(title)}</h1>{sub}</div>",
                unsafe_allow_html=True)


def regime_badge(regime: str | None) -> str:
    if not regime:
        return "<span class='gi-badge gi-muted'>Không phân lớp</span>"
    c = REGIME_COLORS.get(regime, "#8A8680")
    return (f"<span class='gi-badge' style='color:{c}'>"
            f"{REGIME_LABELS.get(regime, regime)} <small>{regime}</small></span>")


def kpi_row(items: list[tuple]):
    """items = [(label, value, delta?, help?)…] → hàng st.metric có viền."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value, *rest = item
        delta = rest[0] if len(rest) > 0 else None
        help_ = rest[1] if len(rest) > 1 else None
        col.metric(label, value, delta, help=help_)


def callout(text: str, kind: str = "info"):
    st.markdown(f"<div class='gi-callout gi-{kind}'>{text}</div>", unsafe_allow_html=True)


def fmt_usd(v, digits: int = 1) -> str:
    return "—" if v is None else f"${v:,.{digits}f}"
