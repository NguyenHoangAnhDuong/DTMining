import pandas as pd
from config import MARKET, MACRO, EVENTS


def load_dataset(path):
    """Đọc Excel/CSV và kiểm tra schema trước khi xử lý."""
    if path.suffix.lower() == '.xlsx':
        data = pd.read_excel(path)
    else:
        data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    required = ['Date'] + MARKET + MACRO + EVENTS
    missing = set(required) - set(data.columns)
    if missing or data.columns.duplicated().any():
        raise ValueError(f'Schema không hợp lệ: {missing}')
    return data
