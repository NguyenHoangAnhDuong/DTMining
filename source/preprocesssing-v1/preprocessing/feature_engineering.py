import numpy as np
import pandas as pd
from config import MARKET, MACRO, EVENTS


def market_returns(data):
    """Log-return = ln(P_t/P_t-1); dầu dùng sai phân vì có thể âm."""
    # Bước 1: tính biến đổi từng chuỗi; không lấy log của giá không dương.
    features = {}
    for column in MARKET:
        series = data[column]
        if column == 'Oil':
            features[column + '_diff'] = series.diff()
        else:
            positive = series.where(series > 0)
            features[column + '_logret'] = np.log(positive).diff()
    return features


def lag_features(data, gold_return):
    """shift(k) chỉ lấy thông tin k quan sát trước thời điểm dự báo."""
    features = {}
    # Giá và return vàng có nhiều độ trễ để khảo sát ngắn/trung hạn.
    for lag in [1, 2, 3, 5, 7, 10, 20, 30]:
        features[f'Gold Price_lag{lag}'] = data['Gold Price'].shift(lag)
        features[f'Gold Price_logret_lag{lag}'] = gold_return.shift(lag)
    for column in ['Silver', 'Copper', 'Oil', 'DJI', 'S&P 500', 'U.S dollar index']:
        for lag in [1, 2, 3, 5]:
            features[f'{column}_lag{lag}'] = data[column].shift(lag)
    return features


def rolling_features(price, gold_return):
    """Rolling chỉ dùng lịch sử đến t-1; std đo độ biến động."""
    features = {}
    for window in [5, 10, 20, 30, 60]:
        # shift trước rolling: không đưa giá t vào cửa sổ lịch sử.
        past_price = price.shift(1).rolling(window)
        past_return = gold_return.shift(1).rolling(window)
        features[f'Gold_rolling_mean_{window}'] = past_price.mean()
        features[f'Gold_rolling_std_{window}'] = past_price.std()
        features[f'Gold_rolling_min_{window}'] = past_price.min()
        features[f'Gold_rolling_max_{window}'] = past_price.max()
        features[f'Return_rolling_mean_{window}'] = past_return.mean()
        features[f'Return_rolling_std_{window}'] = past_return.std()
    return features


def ratio_calendar(data):
    features = {}
    for column in ['Silver', 'Oil', 'U.S dollar index']:
        denominator = data[column].replace(0, np.nan)
        features['Gold_ratio_' + column] = data['Gold Price'] / denominator
    for name in ['dayofweek', 'day', 'month', 'quarter', 'year']:
        features['calendar_' + name] = getattr(data['Date'].dt, name)
    return features


def extended_features(data):
    """Lag1 là giả định khả dụng, KHÔNG phải dữ liệu point-in-time đã xác minh."""
    features = {}
    for column in MACRO:
        available = data[column].shift(1)
        difference = available.diff()
        features[column + '_available_proxy'] = available
        features[column + '_diff'] = difference
        for lag in [1, 2, 3, 5]:
            features[f'{column}_diff_lag{lag}'] = difference.shift(lag)
    for column in EVENTS:
        features[column + '_lag1'] = data[column].shift(1)
    return features


def engineer(data, extended=False):
    features = market_returns(data)
    gold_return = features['Gold Price_logret']
    features.update(lag_features(data, gold_return))
    features.update(rolling_features(data['Gold Price'], gold_return))
    features.update(ratio_calendar(data))
    candidates = MARKET + list(features)
    if extended:
        extra = extended_features(data)
        features.update(extra)
        candidates += list(extra)
    features['Cluster_Return'] = gold_return.rolling(20).mean()
    features['Cluster_Volatility'] = gold_return.rolling(20).std()
    features['Cluster_Current_Return'] = gold_return
    result = pd.concat([data, pd.DataFrame(features, index=data.index)], axis=1)
    return result.replace([np.inf, -np.inf], np.nan), candidates
