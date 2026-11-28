"""Bộ chọn dùng chung — lựa chọn được giữ khi chuyển trang.

Mỗi trang dùng widget key riêng (key@trang) để tránh xung đột trạng thái khi đổi trang;
giá trị dùng chung lưu ở st.session_state["_keep_<key>"] và được nạp lại trước khi tạo widget.
"""
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from source.services.constants import BRANCH_LABELS, HORIZONS, MODEL_NAMES, SPLIT_LABELS


def bind(key: str, default, options=None):
    """Trả (widget_key, on_change) cho một lựa chọn cần giữ giữa các trang."""
    ctx = get_script_run_ctx()
    wkey = f"{key}@{ctx.page_script_hash[:8]}" if ctx and ctx.page_script_hash else key
    store = f"_keep_{key}"
    value = st.session_state.get(store, default)
    if options is not None and value not in options:
        value = default
    st.session_state[wkey] = value

    def _save():
        st.session_state[store] = st.session_state[wkey]

    return wkey, _save


def model_select(models: list[str], key: str = "model", label: str = "Mô hình"):
    wkey, save = bind(key, models[0], models)
    return st.selectbox(label, models, format_func=lambda m: MODEL_NAMES.get(m, m), key=wkey, on_change=save)


def horizon_select(key: str = "horizon", label: str = "Khoảng dự đoán"):
    wkey, save = bind(key, HORIZONS[0], HORIZONS)
    return st.segmented_control(label, HORIZONS, format_func=lambda h: f"h = {h} ngày",
                                key=wkey, required=True, on_change=save)


def branch_select(key: str = "branch", label: str = "Phân lớp giai đoạn trước khi dự đoán?"):
    wkey, save = bind(key, "kmeans", list(BRANCH_LABELS))
    return st.segmented_control(label, list(BRANCH_LABELS), format_func=lambda b: BRANCH_LABELS[b],
                                key=wkey, required=True, on_change=save)


def split_select(key: str = "split", default: str = "test"):
    wkey, save = bind(key, default, list(SPLIT_LABELS))
    return st.segmented_control("Tập đánh giá", list(SPLIT_LABELS), format_func=lambda s: SPLIT_LABELS[s],
                                key=wkey, required=True, on_change=save)
