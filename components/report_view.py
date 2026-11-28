"""Hiển thị báo cáo markdown có hình ảnh (Streamlit không tự đọc ảnh theo đường dẫn tương đối)."""
import re
from pathlib import Path

import streamlit as st

IMAGE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)\s*$")


@st.cache_data(show_spinner=False)
def _blocks(text: str, folder: str) -> list[tuple[str, str, str]]:
    """Tách markdown thành các khối ('md', nội dung) / ('img', đường dẫn, chú thích)."""
    blocks, buf = [], []
    for line in text.splitlines():
        m = IMAGE.match(line.strip())
        if m:
            if buf:
                blocks.append(("md", "\n".join(buf), ""))
                buf = []
            blocks.append(("img", str(Path(folder) / m["src"]), m["alt"]))
        else:
            buf.append(line)
    if buf:
        blocks.append(("md", "\n".join(buf), ""))
    return blocks


def render_markdown(text: str, folder: str):
    for kind, content, alt in _blocks(text, folder):
        if kind == "md":
            st.markdown(content)
        elif Path(content).is_file():
            st.image(content, caption=alt, width="stretch")
        else:
            st.caption(f"(Không tìm thấy hình: {content})")


def headings(text: str) -> list[str]:
    return [l[3:].strip() for l in text.splitlines() if l.startswith("## ")]
