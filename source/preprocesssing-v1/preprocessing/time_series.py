import matplotlib.pyplot as plt
from utils import csv, step
from preprocessing.visualization import finish


def prepare_time_series(data, out):
    """Chỉ giữ ngày trong tuần; không khẳng định đây là lịch giao dịch chính thức."""
    data = data.sort_values('Date')
    weekend = data['Date'].dt.dayofweek >= 5
    csv(data.loc[weekend], out / 'reports/weekends.csv')
    data = data.loc[~weekend].reset_index(drop=True)
    gaps = data[['Date']].copy()
    gaps['calendar_gap'] = gaps['Date'].diff().dt.days
    csv(gaps[gaps['calendar_gap'] >= 4], out / 'reports/date_gaps.csv')
    step(f'Loại cuối tuần: {weekend.sum()}; còn {len(data)} dòng')
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data['Date'], data['Gold Price'])
    ax.set_title('Gold price history (descriptive only)')
    finish(fig, out / 'figures/gold_history.png')
    return data
