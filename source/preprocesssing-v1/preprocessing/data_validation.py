import numpy as np


def validate_parts(parts, horizon):
    target_date = f'target_date_h{horizon}'
    for data in parts:
        if data.empty or not data.Date.is_monotonic_increasing or data.Date.duplicated().any():
            raise ValueError('Tập dữ liệu rỗng hoặc ngày không hợp lệ')
        if not (data[target_date] > data.Date).all():
            raise ValueError('Target không nằm sau origin')
    for earlier, later in zip(parts, parts[1:]):
        if earlier[target_date].max() >= later.Date.min():
            raise ValueError('Target vượt ranh giới tập dữ liệu')


def validate_export(data, features):
    if data.isna().any().any() or not np.isfinite(data.select_dtypes('number')).all().all():
        raise ValueError('NaN hoặc Infinity trong tập xuất')
    if any(c.startswith('target_') or c == 'Date' for c in features):
        raise ValueError('Target/Date trong feature')
    for prefix in ['kmeans_', 'gmm_', 'gmm_prob_']:
        cols = [prefix + r for r in ['Bear', 'Sideways', 'Bull']]
        if not np.allclose(data[cols].sum(axis=1), 1):
            raise ValueError('Nhãn hoặc xác suất cụm không hợp lệ')
