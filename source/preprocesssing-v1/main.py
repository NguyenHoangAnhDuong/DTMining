import argparse
import hashlib
import logging
from pathlib import Path
import pandas as pd
from config import H
from utils import csv, save, step
from preprocessing.data_loader import load_dataset
from preprocessing.data_understanding import explore
from preprocessing.data_cleaning import clean
from preprocessing.time_series import prepare_time_series
from preprocessing.missing_value import handle_missing
from preprocessing.outlier_detection import detect_outliers
from preprocessing.stationarity import scan
from preprocessing.time_series_analysis import analyze
from preprocessing.feature_engineering import engineer
from preprocessing.target_engineering import build_targets
from preprocessing.split import split_horizon
from preprocessing.walk_forward import expanding_folds
from preprocessing.feature_selection import select_features
from preprocessing.normalization import fit_feature_scaler
from preprocessing.clustering import fit_clustering, transform_clusters, task_additions
from preprocessing.academic_reports import domain_check, feature_dictionary, compare_targets
from preprocessing.normalization import export_scaled
from preprocessing.clustering import academic_cluster_experiment, regime_profiles
from preprocessing.data_validation import validate_parts, validate_export


def fit_pack(parts, features, horizon, path, plots, dimensions):
    """Đơn vị thực nghiệm: mọi tham số học chỉ từ parts[0] (train)."""
    validate_parts(parts, horizon)
    path.mkdir(parents=True, exist_ok=True)
    train = parts[0]
    selection = select_features(train, features, horizon, path, plots)
    scaler = fit_feature_scaler(train, features, path)
    bundle = fit_clustering(train, path, plots, dimensions)
    variants = task_additions()
    targets = {'price': f'target_h{horizon}', 'logret': f'target_logret_h{horizon}'}
    save(
        {'selection': selection, 'task_additions': variants, 'targets': targets},
        path / 'features.json'
    )
    columns = [
        'Date'] + features + sum(variants.values(),
        []
    ) + list(targets.values()) + [f'target_date_h{horizon}']
    audit = []
    exported = []
    for name, data in zip(['train', 'val', 'test'], parts):
        result = transform_clusters(data, bundle)[columns]
        validate_export(result, features)
        exported.append(result)
        csv(result, path / f'{name}.csv')
        audit.append({
            'split': name,
            'rows': len(result),
            'first_origin': result.Date.min(),
            'last_target': result[f'target_date_h{horizon}'].max(),
            'status': 'passed'
        })
    csv(pd.DataFrame(audit), path / 'audit.csv')
    if plots:
        export_scaled(exported, features, scaler, path)
        academic_cluster_experiment(train, path)
        regime_profiles(parts, bundle, path)


def run(args):
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise ValueError('Hãy chọn thư mục đầu ra mới hoặc rỗng')
    step('01 - Đọc và khám phá dữ liệu')
    raw = load_dataset(args.input)
    step(f'Dataset: {raw.shape}')
    explore(raw, out, 'raw')
    step('02 - Làm sạch ngày và bản ghi trùng')
    data = clean(raw, out)
    csv(data, out / 'intermediate/01_clean.csv')
    step('03 - Chuỗi thời gian và missing value')
    data = prepare_time_series(data, out)
    data = domain_check(data, out)
    data = handle_missing(data, out)
    csv(data, out / 'intermediate/02_time_missing.csv')
    explore(data, out, 'cleaned')
    train_history = data[data.Date < '2018-01-01']
    step('04 - IQR, ADF/KPSS, ACF/PACF, STL trên lịch sử train')
    detect_outliers(train_history, out, data)
    scan(train_history, out)
    analyze(train_history, out)
    step('05 - Feature engineering và target engineering')
    features_data, features = engineer(data, args.extended)
    csv(features_data, out / 'intermediate/03_features.csv')
    dataset = build_targets(features_data)
    csv(dataset, out / 'base_dataset.csv')
    feature_dictionary(dataset, features, out)
    compare_targets(dataset, out)
    exclusions = []
    report = []
    for horizon in H:
        step(f'06 - H{horizon}: split, selection, scaling, clustering')
        parts = split_horizon(dataset, features, horizon)
        train, val, test = parts
        # Phân loại loại trừ theo thứ tự: warm-up, thiếu feature, thiếu target, purge.
        from preprocessing.split import CLUSTER_COLUMNS
        feature_ok = dataset[features + CLUSTER_COLUMNS].notna().all(axis=1)
        target_ok = dataset[[f'target_h{horizon}', f'target_logret_h{horizon}', f'target_date_h{horizon}']].notna().all(axis=1)
        eligible = feature_ok & target_ok
        exclusions.append(dict(horizon=horizon, total=len(dataset), warmup=int((~feature_ok & (pd.Series(range(len(dataset)), index=dataset.index) < 61)).sum()), feature_missing_after_warmup=int((~feature_ok & (pd.Series(range(len(dataset)), index=dataset.index) >= 61)).sum()), target_missing=int((feature_ok & ~target_ok).sum()), purge=int(eligible.sum()-sum(map(len,parts)))))
        fit_pack(parts, features, horizon, out / f'horizons/h{horizon}', True, args.cluster_dimensions)
        report.append(dict(
            horizon=horizon,
            fold='holdout',
            train=len(train),
            val=len(val),
            test=len(test)
        ))
        for number, fold_train, fold_val in expanding_folds(train, horizon, args.folds):
            step(f'07 - H{horizon}: fold {number}/{args.folds}')
            fit_pack(
                (fold_train, fold_val),
                features,
                horizon,
                out / f'walk_forward/h{horizon}/fold_{number}',
                False,
                args.cluster_dimensions
            )
            report.append(dict(horizon=horizon, fold=number, train=len(fold_train), val=len(fold_val)))
    csv(pd.DataFrame(report), out / 'reports/splits.csv')
    csv(pd.DataFrame(exclusions), out/'reports/exclusions.csv')
    save(
        dict(version='3.2', input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(), extended=args.extended, rows=len(data), features=len(features), folds=args.folds, cluster_dimensions=args.cluster_dimensions, limitations=['Observed rows, not verified trading calendar', 'Market availability unverified', 'Macro/event lag1 is only a proxy', 'Conflicting dates quarantined']),
        out / 'metadata.json'
    )
    step('08 - Hoàn thành')


def main():
    parser = argparse.ArgumentParser(description='Gold preprocessing V3.2')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--cluster-dimensions', type=int, choices=[2, 3], default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
    run(args)


if __name__ == '__main__':
    main()
