import pandas as pd
import matplotlib.pyplot as plt
from utils import csv
from preprocessing.visualization import finish


def explore(data, out, label):
    """EDA mô tả; không dùng thống kê toàn bộ dữ liệu để fit mô hình."""
    from preprocessing.academic_reports import dictionary
    dictionary(data, out, label)
    report = pd.DataFrame({
        'column': data.columns,
        'dtype': data.dtypes.astype(str).values,
        'missing': data.isna().sum().values,
        'unique': data.nunique().values,
    })
    csv(report, out / f'reports/{label}_schema.csv')
    csv(data.describe(include='all').T.reset_index(), out / f'reports/{label}_describe.csv')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(report['column'], report['missing'])
    ax.set_title(f'Missing values: {label}')
    finish(fig, out / f'figures/{label}_missing.png')
