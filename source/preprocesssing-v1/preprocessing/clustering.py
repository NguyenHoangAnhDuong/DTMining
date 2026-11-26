import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from preprocessing.split import CLUSTER_COLUMNS
from preprocessing.visualization import finish
from utils import csv

REGIMES = ['Bear', 'Sideways', 'Bull']


def fit_cluster_models(train, columns):
    """Fit scaler và hai thuật toán trên train; không học từ validation/test."""
    # Chuẩn hóa vì khoảng cách Euclid nhạy với thang đo.
    scaler = StandardScaler().fit(train[columns])
    scaled = scaler.transform(train[columns])
    # K-means++ khởi tạo tâm; thư viện lặp gán cụm và cập nhật centroid.
    kmeans = KMeans(n_clusters=3, n_init=20, random_state=42).fit(scaled)
    # EM ước lượng trọng số, trung bình và hiệp phương sai Gaussian.
    gmm = GaussianMixture(n_components=3, n_init=3, random_state=42).fit(scaled)
    return scaler, scaled, {'kmeans': kmeans, 'gmm': gmm}


def map_regimes(train, labels):
    """Tên cụm là thứ hạng return tương đối, không phải nhãn xu hướng thật."""
    table = pd.DataFrame({'label': labels, 'return': train.Cluster_Return.to_numpy()})
    means = table.groupby('label')['return'].mean().sort_values()
    if len(means) != 3:
        raise ValueError('Không đủ ba cụm để gán regime')
    return {int(label): regime for label, regime in zip(means.index, REGIMES)}


def evaluate_clusters(scaled, labels):
    return {
        'silhouette': silhouette_score(scaled, labels, sample_size=min(1000, len(scaled)), random_state=42),
        'davies_bouldin': davies_bouldin_score(scaled, labels)
    }


def fit_clustering(train, path, plots, dimensions=3):
    columns = CLUSTER_COLUMNS[:dimensions]
    scaler, scaled, models = fit_cluster_models(train, columns)
    mappings = {}
    metrics = []
    summaries = []
    for name, model in models.items():
        labels = model.predict(scaled)
        mappings[name] = map_regimes(train, labels)
        metrics.append(dict(model=name, **evaluate_clusters(scaled, labels)))
        table = train[['Cluster_Return', 'Cluster_Volatility']].copy()
        table['regime'] = [mappings[name][int(label)] for label in labels]
        for regime, group in table.groupby('regime'):
            summaries.append({
                'model': name,
                'regime': regime,
                'count': len(group),
                'mean_return': group.Cluster_Return.mean(),
                'mean_volatility': group.Cluster_Volatility.mean()
            })
        if plots:
            fig, ax = plt.subplots(figsize=(8, 5))
            for regime, group in table.groupby('regime'):
                ax.scatter(group.Cluster_Return, group.Cluster_Volatility, s=8, alpha=.5, label=regime)
            ax.set(xlabel='Rolling return', ylabel='Rolling volatility', title=name)
            ax.legend()
            finish(fig, path / f'figures/{name}_clusters.png')
            fig, ax = plt.subplots(figsize=(12, 4))
            for regime in REGIMES:
                mask = table.regime == regime
                ax.scatter(train.loc[mask, 'Date'], train.loc[mask, 'Gold Price'], s=5, label=regime)
            ax.legend()
            ax.set_title(f'{name}: train price by regime')
            finish(fig, path / f'figures/{name}_price_regime.png')
    csv(pd.DataFrame(metrics), path / 'clustering_metrics.csv')
    csv(pd.DataFrame(summaries), path / 'regime_summary.csv')
    if plots:
        trials = []
        for k in range(2, 7):
            for name, model in [
                ('kmeans', KMeans(n_clusters=k, n_init=10, random_state=42)),
                ('gmm', GaussianMixture(n_components=k, random_state=42))
            ]:
                model.fit(scaled)
                labels = model.predict(scaled)
                if len(set(labels)) > 1:
                    trials.append(dict(model=name, k=k, **evaluate_clusters(scaled, labels)))
        csv(pd.DataFrame(trials), path / 'cluster_k_experiment.csv')
    bundle = dict(scaler=scaler, **models, mapping=mappings, columns=columns)
    joblib.dump(bundle, path / 'clustering.joblib')
    return bundle


def transform_clusters(data, bundle):
    result = data.copy()
    scaled = bundle['scaler'].transform(result[bundle['columns']])
    for name in ['kmeans', 'gmm']:
        labels = bundle[name].predict(scaled)
        for label, regime in bundle['mapping'][name].items():
            result[name + '_' + regime] = (labels == label).astype(int)
    probabilities = bundle['gmm'].predict_proba(scaled)
    for label, regime in bundle['mapping']['gmm'].items():
        result['gmm_prob_' + regime] = probabilities[:, label]
    return result


def task_additions():
    return {
        'task2': [],
        'task1_kmeans': ['kmeans_' + r for r in REGIMES],
        'task1_gmm': ['gmm_' + r for r in REGIMES],
        'task1_gmm_probability': ['gmm_prob_' + r for r in REGIMES]
    }


def academic_cluster_experiment(train, path):
    """So sánh cùng train, 2/3 chiều, K=2..6. Không dùng test chọn K."""
    import numpy as np
    trials = []
    centers = []
    for dimensions in [2,3]:
        columns = CLUSTER_COLUMNS[:dimensions]
        scaler = StandardScaler().fit(train[columns])
        scaled = scaler.transform(train[columns])
        for k in range(2,7):
            models = {'kmeans': KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42), 'gmm': GaussianMixture(n_components=k,n_init=3,random_state=42)}
            for name, model in models.items():
                model.fit(scaled)
                labels = model.predict(scaled)
                row = dict(dimensions=dimensions,k=k,model=name,iterations=model.n_iter_)
                if 1 < len(set(labels)) < len(labels):
                    row.update(evaluate_clusters(scaled,labels))
                if name == 'kmeans':
                    row['sse'] = model.inertia_
                    points = model.cluster_centers_
                else:
                    row.update(aic=model.aic(scaled),bic=model.bic(scaled),converged=model.converged_)
                    points = model.means_
                trials.append(row)
                for label, point in enumerate(scaler.inverse_transform(points)):
                    centers.append(dict(dimensions=dimensions,k=k,model=name,label=label,**dict(zip(columns,point))))
    table = pd.DataFrame(trials)
    csv(table,path/'cluster_dimensions_experiment.csv')
    csv(pd.DataFrame(centers),path/'cluster_centroids.csv')
    for metric in ['sse','silhouette','davies_bouldin','aic','bic']:
        fig, ax = plt.subplots(figsize=(8,4))
        for (name,dimensions), group in table.dropna(subset=[metric]).groupby(['model','dimensions']):
            ax.plot(group.k,group[metric],marker='o',label=f'{name} {dimensions}D')
        ax.set(xlabel='K',ylabel=metric,title=metric+' (train only)')
        ax.legend()
        finish(fig,path/f'figures/cluster_{metric}.png')


def regime_profiles(parts, bundle, path):
    """Mô tả kinh tế: return trung bình, biến động, tỷ lệ return dương.
    Tên Sideways chỉ là thứ hạng tương đối, không phải nhãn đã xác thực.
    """
    rows = []
    for split, data in zip(['train','val','test'],parts):
        result = transform_clusters(data,bundle)
        for model in ['kmeans','gmm']:
            for regime in REGIMES:
                group = result[result[model+'_'+regime]==1]
                rows.append(dict(split=split,model=model,regime=regime,count=len(group),percent=100*len(group)/len(data),mean_return=group.Cluster_Return.mean(),mean_volatility=group.Cluster_Volatility.mean(),positive_return_share=(group.Cluster_Current_Return>0).mean()))
    csv(pd.DataFrame(rows),path/'regime_profiles.csv')
