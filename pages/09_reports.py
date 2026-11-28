import subprocess
import sys

import streamlit as st

from components import loaders
from components.metric_cards import page_header
from components.report_view import headings, render_markdown
from source.services.constants import ROOT_DIR

page_header("Báo cáo", "Báo cáo được sinh từ pipeline. Có thể đọc trực tiếp, tải về hoặc tạo lại.", "Kết quả")

REPORTS = {
    "forecast": ("Kết quả dự đoán: có vs không phân lớp", "result_forecast.md",
                 [sys.executable, "-m", "source.model.report_forecast"]),
    "kmeans": ("Phân cụm giai đoạn thị trường (K-Means)", "result_k_mean.md",
               [sys.executable, "-m", "source.preprocessing_v3.report_kmeans"]),
}

c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
name = c1.segmented_control("Báo cáo", list(REPORTS), format_func=lambda k: REPORTS[k][0],
                            default="forecast", key="report_name", required=True)
title, filename, cmd = REPORTS[name]
text, folder = loaders.report(name)

c2.download_button("Tải báo cáo (.md)", text.encode("utf-8"), filename, "text/markdown",
                   width="stretch", disabled=not text, on_click="ignore")
c3.download_button("Tải toàn bộ kết quả (.zip)", loaders.results_zip(), "results.zip", "application/zip",
                   width="stretch", on_click="ignore", help="Mọi CSV, MD, PNG trong results/ (không gồm file model)")

with st.expander("Tạo lại báo cáo từ kết quả hiện có"):
    st.caption(f"Chạy `{' '.join(cmd[1:])}` — không train lại model. Dùng sau khi có kết quả mới trong `results/`.")
    if st.button("Tạo lại báo cáo", type="primary"):
        with st.status("Đang tạo báo cáo…", expanded=True) as status:
            proc = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8")
            if proc.returncode == 0:
                status.update(label="Đã tạo lại báo cáo", state="complete", expanded=False)
                loaders.clear_all()
                st.rerun()
            else:
                status.update(label="Lỗi khi tạo báo cáo", state="error")
                st.code((proc.stderr or proc.stdout)[-3000:])

if not text:
    st.info(f"Chưa có `{filename}`. Chạy `python -m source.run_pipeline` để sinh báo cáo.")
    st.stop()

sections = headings(text)
left, right = st.columns([1, 4])
with left:
    st.markdown("**Mục lục**")
    for s in sections:
        st.markdown(f"- {s}")
with right, st.container(border=True):
    render_markdown(text, folder)
