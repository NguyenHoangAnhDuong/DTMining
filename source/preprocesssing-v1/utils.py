import json
import logging


def csv(data, path):
    """Xuất bảng trung gian, tự tạo thư mục nếu chưa có."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


def save(data, path):
    """Lưu cấu hình để thành viên train sử dụng đúng danh sách feature."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf8')


def step(message):
    logging.info(message)
