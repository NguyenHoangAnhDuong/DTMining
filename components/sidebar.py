"""Sidebar chung + CSS."""
from pathlib import Path

import streamlit as st

from components import loaders
from source.services.constants import APP_NAME, MODEL_NAMES

CSS = Path(__file__).resolve().parents[1] / "assets" / "styles" / "custom.css"


def _css() -> str:
    # Không cache: file nhỏ, đọc mỗi lần chạy để sửa CSS là thấy ngay (tránh lệch với theme)
    return CSS.read_text(encoding="utf-8") if CSS.is_file() else ""


def inject_css():
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(
            f"<div class='gi-brand'>{APP_NAME}</div>"
            "<div class='gi-brand-sub'>Đồ án Data Mining — phân lớp giai đoạn thị trường "
            "và dự báo giá vàng (XAU/USD)</div>",
            unsafe_allow_html=True,
        )


def render_sidebar_footer():
    with st.sidebar:
        st.divider()
        models = loaders.available_models()
        lines = ["Dữ liệu: EGPBD (processed_v3)"]
        for m in models:
            meta = loaders.metadata(m)
            lines.append(f"{MODEL_NAMES.get(m, m)}: " + (f"cập nhật {meta['created']}" if meta.get("created") else "có kết quả"))
        if not models:
            lines.append("Chưa có mô hình nào có kết quả")
        st.caption("  \n".join(lines))
        if st.button("Nạp lại kết quả", icon=":material/refresh:", width="stretch",
                     help="Xoá cache sau khi chạy lại pipeline (python -m source.run_pipeline)"):
            loaders.clear_all()
            st.rerun()
