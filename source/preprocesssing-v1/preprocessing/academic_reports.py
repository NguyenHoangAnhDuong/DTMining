"""Minh chứng học thuật bổ sung. Không dùng báo cáo mô tả để fit mô hình."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import MARKET, MACRO, EVENTS, H
from utils import csv
from preprocessing.visualization import finish


def dictionary(data, out, label):
    """DM2: phân biệt thuộc tính định lượng, nhị phân và thời gian."""
    rows = []
    for column in data.columns:
        values = pd.to_numeric(data[column], errors='coerce')
        kind = 'temporal' if column == 'Date' else (('binary' if values.dropna().isin([0,1]).all() else 'discrete nonnegative count') if column in EVENTS else 'numeric')
        rows.append(dict(column=column, attribute_type=kind, missing_percent=100*data[column].isna().mean(), numeric_parse_failures=int((values.isna() & data[column].notna()).sum()) if column != 'Date' else 0, skewness=values.skew() if column != 'Date' else np.nan, role='target_original' if column not in ['Date']+MARKET+MACRO+EVENTS else 'input_or_date'))
    csv(pd.DataFrame(rows), out / f'reports/{label}_dictionary.csv')
    events = []
    for column in EVENTS:
        for value, count in data[column].value_counts(dropna=False).items():
            events.append(dict(column=column, value=value, count=count, percent=100*count/len(data)))
    csv(pd.DataFrame(events), out / f'reports/{label}_event_frequency.csv')
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    for ax, column in zip(axes.flat, MARKET+MACRO):
        ax.hist(pd.to_numeric(data[column], errors='coerce').dropna(), bins=30)
        ax.set_title(column)
    for ax in list(axes.flat)[14:]:
        ax.set_visible(False)
    finish(fig, out / f'figures/{label}_histograms.png')


def domain_check(data, out):
    """Giá/chỉ số cần dương để lấy log; dầu và lãi suất có thể âm.
    Event là số đếm nguyên không âm; cách ly giá trị không hợp lệ thành NaN, không suy đoán.
    """
    data = data.copy()
    rows = []
    for column in MARKET + EVENTS:
        if column == 'Oil':
            continue
        invalid = data[column].notna() & (((data[column] < 0) | (data[column] % 1 != 0)) if column in EVENTS else (data[column] <= 0))
        for index in data.index[invalid]:
            rows.append(dict(Date=data.at[index,'Date'], column=column, value=data.at[index,column], action='set_missing'))
        data.loc[invalid, column] = np.nan
    csv(pd.DataFrame(rows, columns=['Date','column','value','action']), out/'reports/domain_violations.csv')
    return data


def feature_dictionary(dataset, features, out):
    """Mỗi feature có công thức và một giá trị thực tế để đối chiếu."""
    rows = []
    for name in features:
        if name in MARKET:
            formula = 'x[t] (assume available at origin)'
        elif name.startswith('calendar_'):
            formula = 'Date[t].' + name.removeprefix('calendar_')
        elif name.startswith('Gold_ratio_'):
            formula = 'Gold[t] / ' + name.removeprefix('Gold_ratio_') + '[t]'
        elif '_rolling_' in name:
            source, _, method, window = name.split('_')
            formula = f'{method}({source}[t-{window}:t-1]), std ddof=1'
        elif '_lag' in name:
            source, lag = name.rsplit('_lag',1)
            formula = f'{source}[t-{lag}]'
        elif name.endswith('_available_proxy'):
            formula = name.removesuffix('_available_proxy') + '[t-1] (unverified release)'
        elif name.endswith('_diff'):
            source = name.removesuffix('_diff')
            formula = f'{source}[t]-{source}[t-1]' if source == 'Oil' else f'{source}[t-1]-{source}[t-2]'
        else:
            source = name.removesuffix('_logret')
            formula = f'ln({source}[t]/{source}[t-1])'
        valid = dataset.loc[dataset[name].notna(), ['Date',name]]
        example = valid.iloc[0] if len(valid) else None
        rows.append(dict(feature=name, formula=formula, example_date=example.Date if example is not None else None, example_value=example[name] if example is not None else None))
    csv(pd.DataFrame(rows), out/'reports/feature_dictionary.csv')


def compare_targets(dataset, out):
    """Chỉ đối chiếu ngày có đúng một target gốc; không dùng target gốc để fit."""
    original = pd.read_csv(out/'reports/original_targets.csv')
    original.Date = pd.to_datetime(original.Date, format='mixed', errors='coerce')
    original = original.dropna(subset=['Date']).drop_duplicates()
    original = original[~original.Date.duplicated(keep=False)]
    joined = dataset.merge(original, on='Date', how='inner')
    rows = []
    for h, name in zip(H, ['Tomorrow gold price','Next week gold price','Next Month gold price']):
        if name not in joined:
            continue
        old = pd.to_numeric(joined[name], errors='coerce')
        new = joined[f'target_h{h}']
        valid = old.notna() & new.notna()
        match = np.isclose(old[valid], new[valid], rtol=0, atol=.01)
        rows.append(dict(horizon=h, compared=int(valid.sum()), matches=int(match.sum()), mae=(old[valid]-new[valid]).abs().mean()))
        csv(pd.DataFrame(dict(Date=joined.Date, original=old, rebuilt=new, difference=old-new)), out/f'reports/target_comparison_h{h}.csv')
    csv(pd.DataFrame(rows), out/'reports/target_comparison_summary.csv')
