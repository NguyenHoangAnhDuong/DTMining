import numpy as np
import pandas as pd
from config import MARKET, MACRO, EVENTS
from utils import csv, step


def handle_missing(data, out):
    """Forward fill tối đa 5 quan sát; không điền Gold Price hoặc event thiếu."""
    data = data.copy()
    original = data.copy()
    before = data.isna().sum()
    data.loc[data['Gold Price'] <= 0, 'Gold Price'] = np.nan
    for column in MARKET[1:] + MACRO:
        data[column] = data[column].ffill(limit=5)
    for column in EVENTS:
        data[column] = data[column].where(data[column] >= 0)
    # Ghi lại từng ô được điền và ngày nguồn gần nhất trong quá khứ.
    changes = []
    for column in MARKET[1:] + MACRO:
        source_dates = original.Date.where(original[column].notna()).ffill()
        filled = original[column].isna() & data[column].notna()
        for index in data.index[filled]:
            changes.append(dict(Date=data.at[index, 'Date'], column=column,
                                filled_value=data.at[index, column], source_date=source_dates.at[index]))
    csv(pd.DataFrame(changes, columns=['Date','column','filled_value','source_date']), out/'reports/missing_details.csv')
    after = data.isna().sum()
    csv(
        pd.DataFrame({'column': before.index, 'before': before.values, 'after': after.values}),
        out / 'reports/missing.csv'
    )
    step(f'Missing trước: {before.sum()}; sau: {after.sum()}')
    return data
