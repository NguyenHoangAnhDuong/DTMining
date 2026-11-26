import numpy as np
import pandas as pd
from config import MARKET, MACRO, EVENTS
from utils import csv, step


def parse_dates(data, out):
    """Ngày không tồn tại được cách ly, không tự suy đoán ngày đúng."""
    data = data.copy()
    data['Date'] = pd.to_datetime(data['Date'], format='mixed', errors='coerce')
    invalid = data['Date'].isna()
    csv(data.loc[invalid], out / 'reports/invalid_dates.csv')
    step(f'Ngày lỗi: {invalid.sum()}')
    return data.loc[~invalid].copy()


def convert_numeric(data):
    """Chuyển dấu '-' và chuỗi không hợp lệ thành NaN."""
    data = data.copy()
    for column in MARKET + MACRO + EVENTS:
        data[column] = pd.to_numeric(data[column], errors='coerce')
        data[column] = data[column].replace([np.inf, -np.inf], np.nan)
    return data


def resolve_duplicates(data, out):
    """Giữ một bản ghi giống nhau; cách ly mọi ngày có giá trị mâu thuẫn."""
    duplicate = data.duplicated(keep='first')
    csv(data.loc[duplicate], out / 'reports/exact_duplicates.csv')
    data = data.loc[~duplicate].copy()
    conflict = data['Date'].duplicated(keep=False)
    csv(data.loc[conflict], out / 'reports/conflicts.csv')
    step(f'Trùng hoàn toàn: {duplicate.sum()}; mâu thuẫn: {conflict.sum()}')
    return data.loc[~conflict].copy()


def clean(data, out):
    original_targets = ['Date', 'Tomorrow gold price', 'Next week gold price', 'Next Month gold price']
    csv(data[[c for c in original_targets if c in data]], out / 'reports/original_targets.csv')
    data = data[['Date'] + MARKET + MACRO + EVENTS].copy()
    data = parse_dates(data, out)
    data = convert_numeric(data)
    return resolve_duplicates(data, out)
