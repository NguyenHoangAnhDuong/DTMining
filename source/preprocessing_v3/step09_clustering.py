"""
Step 09 — Market Regime Clustering với K-Means  (fit ONLY on Train)

Phân lớp giai đoạn thị trường theo 2 trục, đều chỉ dùng quá khứ (shift(1)) → không leakage:
  - Xu hướng  : Gold_logret_roll_mean60  — log-return trung bình 60 phiên trước t
  - Biến động : Gold_logret_roll_std20   — độ lệch chuẩn log-return 20 phiên trước t

Lý do chọn (xem results/kmeans/result_k_mean.md):
  - KHÔNG dùng log-return của chính ngày t: làm cụm bị chia theo "hôm nay tăng/giảm" (nhiễu),
    giai đoạn đổi liên tục (~4.5 ngày/đợt).
  - Xu hướng dài (60) + biến động ngắn (20) cho Silhouette / Davies-Bouldin tốt nhất,
    giai đoạn bền hơn (~14 ngày/đợt) và mỗi cụm có ý nghĩa kinh tế rõ ràng.

Đặt tên cụm (k = 4) theo tâm cụm:
  - Cụm có biến động cao nhất                       → Volatile   (Biến động mạnh / khủng hoảng)
  - 3 cụm còn lại sắp theo xu hướng tăng dần        → Bear (Giảm), Bull (Tăng), StrongBull (Tăng mạnh)

Cột thêm vào dữ liệu:
  - kmeans_<Regime> : one-hot (0/1) cho từng giai đoạn
  - regime_id       : chỉ số giai đoạn theo thứ tự REGIMES (NaN nếu thiếu lịch sử)
  - market_phase    : tên giai đoạn (dễ đọc, dùng cho dashboard)

Model K-Means (.joblib) lưu tại: dataset/processed_v3/models/kmeans.joblib
Web app dùng `assign_regime()` để gán giai đoạn cho dữ liệu mới trước khi dự đoán
(cần ít nhất TREND_WINDOW + 1 phiên lịch sử).
"""
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import KMEANS_MODEL, REPORTS_DIR

logger = logging.getLogger(__name__)

RETURN_COL   = "Gold Price_logret"
TREND_WINDOW = 60
VOL_WINDOW   = 20
TREND_COL    = f"Gold_logret_roll_mean{TREND_WINDOW}"
VOL_COL      = f"Gold_logret_roll_std{VOL_WINDOW}"
CLUSTER_FEATURES = [TREND_COL, VOL_COL]

REGIMES = ["Bear", "Bull", "StrongBull", "Volatile"]
REGIME_LABELS_VI = {
    "Bear":       "Giảm",
    "Bull":       "Tăng",
    "StrongBull": "Tăng mạnh",
    "Volatile":   "Biến động mạnh",
}
N_CLUSTERS = len(REGIMES)


def add_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo feature cho clustering. Chỉ dùng quá khứ nên chạy được trên toàn bộ dữ liệu."""
    df = df.sort_values("Date").reset_index(drop=True).copy()
    past_lr = df[RETURN_COL].shift(1)
    df[TREND_COL] = past_lr.rolling(TREND_WINDOW, min_periods=TREND_WINDOW).mean()
    df[VOL_COL]   = past_lr.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
    return df


def _name_clusters(centers: pd.DataFrame) -> dict:
    """centers: DataFrame (giá trị gốc) index = cluster id, cột = CLUSTER_FEATURES."""
    volatile = int(centers[VOL_COL].idxmax())
    rest = centers.drop(index=volatile).sort_values(TREND_COL).index.tolist()
    mapping = {int(c): r for c, r in zip(rest, ["Bear", "Bull", "StrongBull"])}
    mapping[volatile] = "Volatile"
    return mapping


def _evaluate(X: np.ndarray, labels: np.ndarray) -> dict:
    n = min(2000, len(X))
    return {
        "silhouette"    : float(silhouette_score(X, labels, sample_size=n, random_state=42)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def fit_on_train(train: pd.DataFrame) -> dict:
    valid  = train[CLUSTER_FEATURES].dropna()
    scaler = StandardScaler().fit(valid)
    X      = scaler.transform(valid)
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=42).fit(X)

    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=CLUSTER_FEATURES)
    mapping = _name_clusters(centers)

    metrics = {"model": "kmeans", "k": N_CLUSTERS, **_evaluate(X, kmeans.labels_)}
    logger.info("K-Means — Silhouette: %.4f  Davies-Bouldin: %.4f",
                metrics["silhouette"], metrics["davies_bouldin"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "clustering_metrics.csv", index=False)

    bundle = dict(scaler=scaler, kmeans=kmeans, mapping=mapping, columns=CLUSTER_FEATURES,
                  trend_window=TREND_WINDOW, vol_window=VOL_WINDOW, regimes=REGIMES)
    KMEANS_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, KMEANS_MODEL)
    logger.info("K-Means model saved → %s  | mapping: %s", KMEANS_MODEL, mapping)
    return bundle


def transform(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Gán nhãn giai đoạn cho df đã có CLUSTER_FEATURES."""
    df   = df.copy()
    cols = bundle["columns"]

    valid_idx = df.index[df[cols].notna().all(axis=1)]
    labels    = bundle["kmeans"].predict(bundle["scaler"].transform(df.loc[valid_idx, cols]))
    regimes   = pd.Series(labels, index=valid_idx).map(bundle["mapping"])

    for regime in REGIMES:
        df[f"kmeans_{regime}"] = (regimes.reindex(df.index) == regime).astype(int)
    df["market_phase"] = regimes.reindex(df.index)
    df["regime_id"]    = df["market_phase"].map({r: i for i, r in enumerate(REGIMES)})

    logger.info("K-Means transformed: %d / %d rows assigned", len(valid_idx), len(df))
    return df


def assign_regime(history: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Dùng cho web app: nhận dữ liệu dạng base_dataset (có lịch sử ≥ TREND_WINDOW+1 phiên),
    trả về dữ liệu đã gán giai đoạn thị trường."""
    if bundle is None:
        bundle = joblib.load(KMEANS_MODEL)
    return transform(add_cluster_features(history), bundle)


def run(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """train/val/test phải đã có CLUSTER_FEATURES (gọi add_cluster_features trước khi split)."""
    logger.info("Step 09 — K-Means clustering (fit on train only)")
    bundle = fit_on_train(train)
    return transform(train, bundle), transform(val, bundle), transform(test, bundle), bundle
