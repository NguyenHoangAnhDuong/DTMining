from sklearn.model_selection import TimeSeriesSplit


def expanding_folds(train, horizon, folds):
    """Mỗi fold có train mở rộng và validation tương lai; purge lại nhãn."""
    target_date = f'target_date_h{horizon}'
    splitter = TimeSeriesSplit(n_splits=folds)
    for number, (train_indices, val_indices) in enumerate(splitter.split(train), 1):
        fold_train = train.iloc[train_indices]
        fold_val = train.iloc[val_indices]
        fold_train = fold_train[fold_train[target_date] < fold_val.Date.min()]
        yield number, fold_train, fold_val
