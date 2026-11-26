import pandas as pd
import matplotlib.pyplot as plt
from config import MARKET, MACRO
from utils import csv
from preprocessing.visualization import finish


def detect_outliers(train, out, full=None):
    """IQR = Q3 - Q1; ngoại lai nằm ngoài Q1-1.5*IQR và Q3+1.5*IQR.
    Chỉ phát hiện, không capping vì biến động cực đoan có thể là sự kiện thật.
    """
    rows = []
    details = []
    full = train if full is None else full
    for column in MARKET + MACRO:
        values = train[column].dropna()
        q1 = values.quantile(.25)
        q3 = values.quantile(.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (values < lower) | (values > upper)
        extreme_lower = q1 - 3 * iqr
        extreme_upper = q3 + 3 * iqr
        # Dùng cùng ngưỡng train để mô tả mọi tập, không sửa giá trị.
        for index in full.index[(full[column] < lower) | (full[column] > upper)]:
            value = full.at[index, column]
            date = full.at[index, 'Date']
            split = 'train' if date < pd.Timestamp('2018-01-01') else ('val' if date < pd.Timestamp('2020-01-01') else 'test')
            details.append(dict(Date=date, feature=column, value=value, split=split,
                                severity='extreme' if value < extreme_lower or value > extreme_upper else 'outlier'))
        rows.append({
            'extreme_lower': extreme_lower, 'extreme_upper': extreme_upper,
            'feature': column,
            'q1': q1,
            'q3': q3,
            'lower': lower,
            'upper': upper,
            'train_outliers': int(mask.sum())
        })
    csv(pd.DataFrame(rows), out / 'reports/outliers.csv')
    csv(pd.DataFrame(details, columns=['Date','feature','value','split','severity']), out/'reports/outlier_details.csv')
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    for ax, column in zip(axes.flat, MARKET + MACRO):
        ax.boxplot(train[column].dropna())
        ax.set_title(column)
    for ax in list(axes.flat)[14:]:
        ax.set_visible(False)
    finish(fig, out / 'figures/train_boxplots.png')
