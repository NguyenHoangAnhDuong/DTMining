"""
Step 09 — Market Regime Clustering  (fit ONLY on Train)

Thuật toán: K-Means (k=3) VÀ GMM (k=3)
Features clustering: Gold Price_logret, Gold_logret_roll_mean20, Gold_logret_roll_std20

Nhãn chế độ thị trường được sắp xếp theo rolling mean return:
  Thấp nhất → Bear | Giữa → Sideways | Cao nhất → Bull

Output thêm vào mỗi split:
  - kmeans_Bear, kmeans_Sideways, kmeans_Bull  : one-hot (0/1)
  - gmm_Bear, gmm_Sideways, gmm_Bull           : one-hot (0/1)
  - gmm_prob_Bear, gmm_prob_Sideways, gmm_prob_Bull : xác suất mềm từ GMM (0.0–1.0)
  - market_phase                                : nhãn chuỗi từ K-Means (đọc được)

Bundle clustering (.joblib) lưu tại: dataset/processed_v3/models/clustering.joblib
"""
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import MODELS_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)

CLUSTER_FEATURES = [
    "Gold Price_logret",
    "Gold_logret_roll_mean20",
    "Gold_logret_roll_std20",
]
REGIMES    = ["Bear", "Sideways", "Bull"]
N_CLUSTERS = 3


def _fit(train: pd.DataFrame, cols: list[str]) -> tuple:
    valid  = train[cols].dropna()
    scaler = StandardScaler().fit(valid)
    X      = scaler.transform(valid)
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=42).fit(X)
    gmm    = GaussianMixture(n_components=N_CLUSTERS, n_init=3, random_state=42).fit(X)
    return scaler, X, valid.index, kmeans, gmm


def _map_regimes(train: pd.DataFrame, labels: np.ndarray, valid_idx, return_col: str) -> dict:
    table = pd.DataFrame({"label": labels, "ret": train.loc[valid_idx, return_col].to_numpy()})
    order = table.groupby("label")["ret"].mean().sort_values().index.tolist()
    return {int(lbl): regime for lbl, regime in zip(order, REGIMES)}


def _evaluate(X: np.ndarray, labels: np.ndarray) -> dict:
    n = min(1000, len(X))
    return {
        "silhouette"    : float(silhouette_score(X, labels, sample_size=n, random_state=42)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def fit_on_train(train: pd.DataFrame) -> dict:
    cols = [c for c in CLUSTER_FEATURES if c in train.columns]
    if len(cols) < 2:
        logger.warning("Not enough clustering columns found (%s), skipping.", cols)
        return {}

    scaler, X, valid_idx, kmeans, gmm = _fit(train, cols)
    mappings = {}
    metrics  = []

    for name, model in [("kmeans", kmeans), ("gmm", gmm)]:
        labels = model.predict(X)
        mappings[name] = _map_regimes(train, labels, valid_idx, cols[1])
        m = _evaluate(X, labels)
        metrics.append({"model": name, **m})
        logger.info("%s — Silhouette: %.4f  Davies-Bouldin: %.4f",
                    name.upper(), m["silhouette"], m["davies_bouldin"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(REPORTS_DIR / "clustering_metrics.csv", index=False)

    bundle = dict(scaler=scaler, kmeans=kmeans, gmm=gmm, mapping=mappings, columns=cols)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODELS_DIR / "clustering.joblib")
    logger.info("Clustering bundle saved → %s", MODELS_DIR / "clustering.joblib")
    return bundle


def transform(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    if not bundle:
        return df
    df   = df.copy()
    cols = bundle["columns"]

    valid_mask = df[cols].notna().all(axis=1)
    valid_idx  = df.index[valid_mask]
    X = bundle["scaler"].transform(df.loc[valid_idx, cols])

    # K-Means one-hot
    km_labels = bundle["kmeans"].predict(X)
    for lbl, regime in bundle["mapping"]["kmeans"].items():
        col = f"kmeans_{regime}"
        df[col] = 0
        df.loc[valid_idx[km_labels == lbl], col] = 1

    # GMM one-hot
    gm_labels = bundle["gmm"].predict(X)
    for lbl, regime in bundle["mapping"]["gmm"].items():
        col = f"gmm_{regime}"
        df[col] = 0
        df.loc[valid_idx[gm_labels == lbl], col] = 1

    # GMM soft probabilities
    gm_proba = bundle["gmm"].predict_proba(X)
    for lbl, regime in bundle["mapping"]["gmm"].items():
        col = f"gmm_prob_{regime}"
        df[col] = np.nan
        df.loc[valid_idx, col] = gm_proba[:, lbl]

    # String label từ K-Means (dễ đọc, dùng cho EDA / dashboard)
    df["market_phase"] = pd.NA
    for lbl, regime in bundle["mapping"]["kmeans"].items():
        df.loc[valid_idx[km_labels == lbl], "market_phase"] = regime
    df["market_phase"] = df["market_phase"].ffill()

    logger.info("Clustering transformed: %d / %d rows assigned", len(valid_idx), len(df))
    return df


def run(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    logger.info("Step 09 — Clustering (K-Means + GMM, fit on train only)")
    bundle    = fit_on_train(train)
    train_out = transform(train, bundle)
    val_out   = transform(val,   bundle)
    test_out  = transform(test,  bundle)
    return train_out, val_out, test_out, bundle
