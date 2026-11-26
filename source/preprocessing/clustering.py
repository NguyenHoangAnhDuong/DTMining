# Bước 8 — Market Phase Clustering
# Mục tiêu: gán nhãn giai đoạn thị trường cho mỗi ngày giao dịch
#   Tang (Bull) | Di ngang (Sideways) | Giam (Bear)
# Quy trình:
#   1. Xây dựng 3 features mô tả trạng thái thị trường từ Gold Price_logret:
#        logret          : log-return hàng ngày (momentum tức thời)
#        roll_mean20     : rolling mean 20 ngày   (xu hướng ngắn hạn)
#        roll_std20      : rolling std 20 ngày    (biến động)
#   2. Chuẩn hóa 3 features bằng StandardScaler (chỉ dùng cho clustering)
#   3. Fit K-Means(k=3, n_init=20)
#   4. Đánh giá: Silhouette Score (cao hơn tốt hơn), Davies-Bouldin (thấp hơn tốt hơn)
#   5. Gán nhãn: sort cluster theo roll_mean20 → thấp nhất = Giam, cao nhất = Tang
#   6. Thêm cột market_phase vào dataset, ffill 19 hàng warmup đầu
# Lưu ý:
#   - rolling KHÔNG dùng shift(1) ở đây vì mục đích là mô tả trạng thái, không phải dự báo
#   - scaler chỉ dùng nội bộ bước này, không lưu ra file

import logging
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

INPUT_PATH   = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\after_feature_selection.csv"
OUTPUT_PATH  = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\after_clustering.csv"
REPORT_PATH  = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\reports\clustering_report.csv"

N_CLUSTERS  = 3
ROLL_WINDOW = 20


def _build_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    logret = df["Gold Price_logret"]
    return pd.DataFrame(
        {
            "logret"     : logret.values,
            "roll_mean20": logret.rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).mean().values,
            "roll_std20" : logret.rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).std().values,
        },
        index=df.index,
    )


def _evaluate(X: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    logger.info("K-Means  Silhouette=%.4f | Davies-Bouldin=%.4f", sil, dbi)
    return sil, dbi


def _assign_phase_labels(feat_df: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    # Sắp xếp cluster theo roll_mean20 trung bình: thấp→Giam, giữa→Di ngang, cao→Tang
    temp = feat_df.copy()
    temp["cluster"] = labels
    order = temp.groupby("cluster")["roll_mean20"].mean().sort_values().index.tolist()
    phase_names = ["Giam", "Di ngang", "Tang"]
    phase_map = {cluster_id: name for cluster_id, name in zip(order, phase_names)}
    return pd.Series([phase_map[lbl] for lbl in labels], index=feat_df.index)


def process_clustering(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    feat_df  = _build_cluster_features(df)
    valid_idx = feat_df.dropna().index
    feat_valid = feat_df.loc[valid_idx].values  # numpy array cho sklearn

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feat_valid)

    km     = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    sil, dbi = _evaluate(X_scaled, labels)

    # Gán nhãn pha
    feat_df_valid = feat_df.loc[valid_idx]
    phases = _assign_phase_labels(feat_df_valid, labels)

    df["market_phase"] = pd.Series(dtype="object")
    df.loc[valid_idx, "market_phase"] = phases.values
    # ffill để lấp 19 hàng warmup đầu (rolling chưa đủ 20 điểm)
    df["market_phase"] = df["market_phase"].ffill()

    report = pd.DataFrame(
        {
            "Model"         : ["K-Means"],
            "Silhouette"    : [round(sil, 4)],
            "Davies_Bouldin": [round(dbi, 4)],
        }
    )

    return df, report


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    df = pd.read_csv(INPUT_PATH)
    logger.info("Doc du lieu: %d hang x %d cot", *df.shape)

    df_out, report = process_clustering(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")

    phase_counts = df_out["market_phase"].value_counts()

    print("\n=== Tom tat Buoc 8 — Market Phase Clustering ===")
    print(f"  Mo hinh              : K-Means (k=3)")
    print(f"  So hang              : {len(df_out)}")
    print(f"  Phan phoi giai doan  :")
    for phase, cnt in phase_counts.items():
        print(f"    {phase:<12}: {cnt:>5} ngay ({cnt/len(df_out)*100:.1f}%)")
    print(f"  Bao cao luu tai      : {REPORT_PATH}")
    print(f"  Du lieu luu tai      : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
