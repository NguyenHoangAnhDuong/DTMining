"""Sidebar chung + CSS."""
from pathlib import Path

import streamlit as st

from components import loaders
from source.services.constants import APP_NAME, MODEL_NAMES

CSS = Path(__file__).resolve().parents[1] / "assets" / "styles" / "custom.css"


@st.cache_resource
def _css() -> str:
    return CSS.read_text(encoding="utf-8") if CSS.is_file() else ""


def inject_css():
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(
            f"<div class='gi-brand'><span>◆</span> {APP_NAME.upper()}</div>"
            "<div class='gi-brand-sub'>Giai đoạn thị trường &amp; dự đoán giá vàng</div>",
            unsafe_allow_html=True,
        )


def render_sidebar_footer():
    with st.sidebar:
        st.divider()
        models = loaders.available_models()
        st.caption("Mô hình có kết quả: " + (", ".join(MODEL_NAMES.get(m, m) for m in models) or "—"))
        for m in models:
            meta = loaders.metadata(m)
            if meta.get("created"):
                st.caption(f"{MODEL_NAMES.get(m, m)} cập nhật: {meta['created']}")
        if st.button("↻ Nạp lại kết quả", width="stretch",
                     help="Xoá cache sau khi chạy lại pipeline (python -m source.run_pipeline)"):
            loaders.clear_all()
            st.rerun()
        st.caption("Dataset: EGPBD · processed_v3")
