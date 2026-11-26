import joblib
from sklearn.preprocessing import StandardScaler


def fit_feature_scaler(train, features, path):
    """z=(x-mean_train)/std_train. CSV giữ giá gốc; LSTM dùng scaler riêng.
    Nếu dùng subset hoặc thêm cluster features, fit scaler đúng subset trên train.
    """
    scaler = StandardScaler()
    scaler.fit(train[features])
    joblib.dump(scaler, path / 'feature_scaler.joblib')
    return scaler


def export_scaled(parts, features, scaler, path):
    """Transform bằng mean/std train; giữ nguyên target và nhãn cụm."""
    import pandas as pd
    from utils import csv
    rows = []
    for name, data in zip(['train','val','test'], parts):
        scaled = pd.DataFrame(scaler.transform(data[features]), columns=features, index=data.index)
        result = data.copy()
        result[features] = scaled
        csv(result, path / f'{name}_scaled.csv')
        for column in features:
            rows.append(dict(split=name, feature=column, mean_before=data[column].mean(), std_before=data[column].std(ddof=0), mean_after=scaled[column].mean(), std_after=scaled[column].std(ddof=0)))
    csv(pd.DataFrame(rows), path/'scaling_summary.csv')
