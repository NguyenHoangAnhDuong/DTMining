import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from config import EVENTS
from utils import csv
from preprocessing.visualization import finish


def rank_features(train, features, target):
    """MI đo phụ thuộc thống kê, không chứng minh quan hệ nhân quả."""
    discrete = [c in [e + '_lag1' for e in EVENTS] or c.startswith('calendar_') for c in features]
    # Chỉ dùng X và y của train; event là số đếm rời rạc.
    scores = mutual_info_regression(
        train[features],
        train[target],
        discrete_features=discrete,
        random_state=42
    )
    return pd.Series(scores, index=features).sort_values(ascending=False)


def correlation_filter(train, ranking, correlation, threshold=.95):
    """Ưu tiên MI cao; bỏ cột hằng và cột tương quan mạnh với cột đã giữ."""
    kept = []
    reasons = {}
    for column in ranking.index:
        if train[column].nunique() <= 1:
            reasons[column] = 'constant'
            continue
        # So sánh với biến đã giữ có MI cao hơn hoặc bằng.
        redundant = [other for other in kept if correlation.loc[column, other] > threshold]
        if redundant:
            reasons[column] = 'correlated_with:' + redundant[0]
        else:
            kept.append(column)
            reasons[column] = 'kept'
    return kept, reasons


def select_features(train, features, horizon, path, plots):
    correlation = train[features].corr().abs()
    summary = []
    config = {}
    for name, target in [('price', f'target_h{horizon}'), ('logret', f'target_logret_h{horizon}')]:
        ranking = rank_features(train, features, target)
        kept, reasons = correlation_filter(train, ranking, correlation)
        report = pd.DataFrame({
            'feature': ranking.index,
            'mi': ranking.values,
            'kept_after_corr': ranking.index.isin(kept),
            'reason': [reasons[c] for c in ranking.index]
        })
        csv(report, path / f'mi_{name}.csv')
        config[name] = {str(n): kept[:n] for n in [20, 40, 60, 80]}
        config[name]['all_filtered'] = kept
        config[name]['all_base'] = features
        summary.append(dict(target=name, candidates=len(features), retained=len(kept), removed=len(features)-len(kept), reduction_percent=100*(1-len(kept)/len(features)), zero_mi_retained=sum(ranking[c]==0 for c in kept), **{f'actual_top_{n}':len(kept[:n]) for n in [20,40,60,80]}))
        if plots:
            fig, ax = plt.subplots(figsize=(9, 7))
            ranking.head(20).sort_values().plot.barh(ax=ax)
            ax.set_title(f'Mutual information: {name}')
            finish(fig, path / f'figures/mi_{name}.png')
    csv(pd.DataFrame(summary), path / "feature_reduction.csv")
    if plots:
        correlation.to_csv(path / "correlation_matrix.csv")
        fig, ax = plt.subplots(figsize=(24, 24))
        image = ax.imshow(correlation, vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(features)), features, rotation=90, fontsize=5)
        ax.set_yticks(range(len(features)), features, fontsize=5)
        ax.set_title('Absolute Pearson correlation (train only)')
        fig.colorbar(image, ax=ax)
        finish(fig, path / 'figures/correlation.png')
    return config
