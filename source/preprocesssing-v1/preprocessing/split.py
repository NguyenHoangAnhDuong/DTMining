import pandas as pd

CLUSTER_COLUMNS = ['Cluster_Return', 'Cluster_Volatility', 'Cluster_Current_Return']


def split_horizon(data, features, horizon):
    """Chia theo ngày origin và purge theo ngày target, không shuffle."""
    target_date = f'target_date_h{horizon}'
    required = features + CLUSTER_COLUMNS + [
        f'target_h{horizon}',
        f'target_logret_h{horizon}',
        target_date
    ]
    eligible = data.dropna(subset=required)
    val_start = pd.Timestamp('2018-01-01')
    test_start = pd.Timestamp('2020-01-01')
    train = eligible[(eligible.Date < val_start) & (eligible[target_date] < val_start)]
    val = eligible[(eligible.Date >= val_start) & (eligible.Date < test_start) & (eligible[target_date] < test_start)]
    test = eligible[eligible.Date >= test_start]
    return train, val, test
