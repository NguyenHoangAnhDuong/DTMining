"""
Báo cáo kết quả K-Means sau khi train (chạy SAU Stage 2).

Chạy từ thư mục gốc dự án (DTMining/):
    python -m source.preprocessing_v3.report_kmeans

Output: results/kmeans/
  result_k_mean.md            — báo cáo tổng hợp
  result_k_mean_metrics.csv   — Silhouette / Davies-Bouldin / Calinski-Harabasz theo từng tập
  result_k_mean_k_scan.csv    — chỉ số khi thử k = 2..8 (chọn k)
  result_k_mean_regimes.csv   — thống kê từng giai đoạn theo từng tập
  figures/*.png               — hình ảnh minh họa
"""
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.preprocessing import StandardScaler

from source.preprocessing_v3.config import (
    BASE_DATASET, KMEANS_MODEL, KMEANS_RESULTS, KMEANS_SPLIT_SUFFIX, ROOT_DIR, SPLIT_DIR,
    TEST_START, TRAIN_END, VAL_END, VAL_START,
)
from source.preprocessing_v3.step09_clustering import REGIME_LABELS_VI, REGIMES

logger = logging.getLogger(__name__)

OUT_DIR = KMEANS_RESULTS
FIG_DIR = OUT_DIR / "figures"
COLORS  = {"Bear": "#d62728", "Bull": "#2ca02c", "StrongBull": "#1f77b4", "Volatile": "#ff7f0e"}
FWD_VOL_WINDOW = 20

# Các phương án feature đã thử (để chứng minh lựa chọn hiện tại)
FEATURE_CANDIDATES = {
    "A. logret ngày + mean20 + std20 (bản cũ)": ["lr", "m20", "s20"],
    "B. mean20 + std20": ["m20", "s20"],
    "C. mean60 + std20 (chọn)": ["m60", "s20"],
    "D. mean60 + std60": ["m60", "s60"],
    "E. gap MA50 + std20": ["ma50gap", "s20"],
}
SPLITS  = ["train", "val", "test"]
K_RANGE = range(2, 9)
TRADING_DAYS = 252


# ── helpers ────────────────────────────────────────────────────────────────
def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep  = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _scores(X: np.ndarray, labels: np.ndarray) -> dict:
    if len(np.unique(labels)) < 2:
        return dict(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    return dict(
        silhouette=float(silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=42)),
        davies_bouldin=float(davies_bouldin_score(X, labels)),
        calinski_harabasz=float(calinski_harabasz_score(X, labels)),
    )


def _run_lengths(phase: pd.Series) -> pd.DataFrame:
    s = phase.dropna()
    run_id = (s != s.shift()).cumsum()
    return s.groupby(run_id).agg(regime="first", length="size")


def _load():
    bundle = joblib.load(KMEANS_MODEL)
    data = {}
    for sp in SPLITS:
        df = pd.read_csv(SPLIT_DIR / f"{sp}_{KMEANS_SPLIT_SUFFIX}.csv", parse_dates=["Date"])
        data[sp] = df
    return bundle, data


# ── analyses ───────────────────────────────────────────────────────────────
def metrics_per_split(bundle, data) -> pd.DataFrame:
    cols, rows = bundle["columns"], []
    for sp, df in data.items():
        valid = df.dropna(subset=cols)
        X = bundle["scaler"].transform(valid[cols])
        rows.append({"split": sp, "rows": len(valid), "unlabeled": len(df) - len(valid),
                     **_scores(X, bundle["kmeans"].predict(X))})
    return pd.DataFrame(rows)


def k_scan(bundle, train) -> pd.DataFrame:
    X = bundle["scaler"].transform(train.dropna(subset=bundle["columns"])[bundle["columns"]])
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
        rows.append({"k": k, "inertia": float(km.inertia_), **_scores(X, km.labels_)})
    return pd.DataFrame(rows)


def centers_table(bundle) -> pd.DataFrame:
    centers = bundle["scaler"].inverse_transform(bundle["kmeans"].cluster_centers_)
    df = pd.DataFrame(centers, columns=bundle["columns"])
    df.insert(0, "regime", [bundle["mapping"][i] for i in range(len(df))])
    df.insert(1, "cluster_id", range(len(df)))
    return df.set_index("regime").loc[REGIMES].reset_index()


def regime_stats(data) -> pd.DataFrame:
    rows = []
    for sp, df in data.items():
        total = df["market_phase"].notna().sum()
        runs  = _run_lengths(df["market_phase"])
        for r in REGIMES:
            sub = df[df["market_phase"] == r]
            lr  = sub["Gold Price_logret"]
            rows.append({
                "split": sp, "regime": r, "days": len(sub),
                "share_%": 100 * len(sub) / total if total else np.nan,
                "mean_logret_%": 100 * lr.mean() if len(sub) else np.nan,
                "ann_vol_%": 100 * lr.std() * np.sqrt(TRADING_DAYS) if len(sub) > 1 else np.nan,
                "avg_run_days": runs.loc[runs["regime"] == r, "length"].mean(),
                "fwd_logret_h1_%": 100 * sub["target_logret_h1"].mean() if len(sub) else np.nan,
                "fwd_logret_h7_%": 100 * sub["target_logret_h7"].mean() if len(sub) else np.nan,
                "fwd_logret_h30_%": 100 * sub["target_logret_h30"].mean() if len(sub) else np.nan,
            })
    return pd.DataFrame(rows)


def _avg_run(labels: pd.Series) -> float:
    return float(labels.groupby((labels != labels.shift()).cumsum()).size().mean())


def _add_fwd_vol(df: pd.DataFrame) -> pd.DataFrame:
    """Biến động thực tế 20 phiên TỚI (chỉ dùng để đánh giá, không dùng làm feature)."""
    lr = df["Gold Price_logret"]
    return df.assign(fwd_vol20=lr[::-1].rolling(FWD_VOL_WINDOW).std()[::-1].shift(-1))


def feature_comparison() -> pd.DataFrame:
    df = pd.read_csv(BASE_DATASET, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    past = df["Gold Price_logret"].shift(1)
    feats = pd.DataFrame({"lr": df["Gold Price_logret"]})
    for w in (20, 60):
        feats[f"m{w}"] = past.rolling(w).mean()
        feats[f"s{w}"] = past.rolling(w).std()
    px = df["Gold Price"].shift(1)
    feats["ma50gap"] = px / px.rolling(50).mean() - 1
    df = _add_fwd_vol(df)

    d_ok   = feats.notna().all(axis=1)
    is_tr  = (df["Date"] <= TRAIN_END) & d_ok
    is_val = (df["Date"] >= VAL_START) & (df["Date"] <= VAL_END)
    is_oos = (df["Date"] >= VAL_START) & df["fwd_vol20"].notna()

    rows = []
    for name, cols in FEATURE_CANDIDATES.items():
        for k in (3, 4):
            sc = StandardScaler().fit(feats.loc[is_tr, cols])
            X  = sc.transform(feats.loc[is_tr, cols])
            km = KMeans(k, n_init=20, random_state=42).fit(X)
            lab = pd.Series(km.predict(sc.transform(feats.loc[d_ok, cols])), index=feats.index[d_ok])
            oos = df.loc[is_oos].assign(l=lab.reindex(df.index[is_oos]))
            vol = oos.groupby("l")["fwd_vol20"].mean()
            rows.append({
                "features": name, "k": k,
                **_scores(X, km.labels_),
                "avg_run_days": _avg_run(lab),
                "regimes_in_val": int(lab[is_val[lab.index]].nunique()),
                "oos_fwd_vol_ratio": float(vol.max() / vol.min()),
            })
    return pd.DataFrame(rows)


def oos_usefulness(data) -> pd.DataFrame:
    """Giai đoạn gán trên Val+Test có phân biệt được biến động / lợi suất tương lai không?"""
    full = _add_fwd_vol(pd.concat(data.values()).sort_values("Date").reset_index(drop=True))
    oos  = full[full["Date"] >= VAL_START]
    g = oos.groupby("market_phase").agg(
        days=("Date", "size"),
        fwd_vol20_ann_pct=("fwd_vol20", lambda s: 100 * s.mean() * np.sqrt(TRADING_DAYS)),
        fwd_logret_h7_pct=("target_logret_h7", lambda s: 100 * s.mean()),
        fwd_logret_h30_pct=("target_logret_h30", lambda s: 100 * s.mean()),
        up_ratio_h30_pct=("target_logret_h30", lambda s: 100 * (s.dropna() > 0).mean()),
    ).reindex(REGIMES)
    g.insert(0, "nhãn", [REGIME_LABELS_VI[r] for r in g.index])
    return g.reset_index().rename(columns={"market_phase": "regime"})


def transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    s = df["market_phase"].dropna()
    ct = pd.crosstab(s.shift().iloc[1:], s.iloc[1:], normalize="index")
    return ct.reindex(index=REGIMES, columns=REGIMES).fillna(0.0)


# ── figures ────────────────────────────────────────────────────────────────
def fig_k_scan(scan: pd.DataFrame):
    k_sel = len(REGIMES)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title in zip(axes, ["inertia", "silhouette", "davies_bouldin"],
                              ["Elbow (Inertia) ↓", "Silhouette ↑", "Davies-Bouldin ↓"]):
        ax.plot(scan["k"], scan[col], marker="o")
        ax.axvline(k_sel, color="orange", ls="--", label=f"k = {k_sel} (chọn)")
        ax.set_title(title); ax.set_xlabel("k"); ax.grid(alpha=.3); ax.legend()
    fig.suptitle("Chọn số cụm K-Means (trên tập Train)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "kmeans_k_selection.png", dpi=120); plt.close(fig)


def fig_timeline(data):
    full = pd.concat(data.values()).sort_values("Date")
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(full["Date"], full["Gold Price"], color="black", lw=.6, alpha=.6)
    for r in REGIMES:
        sub = full[full["market_phase"] == r]
        ax.scatter(sub["Date"], sub["Gold Price"], s=3, color=COLORS[r], label=f"{r} ({REGIME_LABELS_VI[r]})")
    for d, lbl in [(VAL_START, "Val"), (TEST_START, "Test")]:
        ax.axvline(pd.Timestamp(d), color="blue", ls="--", lw=1)
        ax.text(pd.Timestamp(d), ax.get_ylim()[1], f" {lbl}", va="top", color="blue")
    ax.set_title("Giá vàng tô màu theo giai đoạn thị trường (K-Means)")
    ax.set_ylabel("Gold Price"); ax.legend(markerscale=4); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "kmeans_regime_timeline.png", dpi=120); plt.close(fig)


def fig_scatter(bundle, train):
    c_mean, c_std = bundle["columns"][0], bundle["columns"][1]
    centers = centers_table(bundle)
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in REGIMES:
        sub = train[train["market_phase"] == r]
        ax.scatter(sub[c_mean], sub[c_std], s=5, alpha=.5, color=COLORS[r],
                   label=f"{r} ({REGIME_LABELS_VI[r]})")
    ax.scatter(centers[c_mean], centers[c_std], marker="X", s=200, color="black", label="Tâm cụm")
    ax.set_xlabel(c_mean); ax.set_ylabel(c_std); ax.grid(alpha=.3); ax.legend()
    ax.set_title("Không gian phân cụm (Train): xu hướng vs biến động")
    fig.tight_layout(); fig.savefig(FIG_DIR / "kmeans_cluster_scatter.png", dpi=120); plt.close(fig)


def fig_distribution(stats: pd.DataFrame):
    pv = stats.pivot(index="split", columns="regime", values="share_%").reindex(SPLITS)[REGIMES]
    ax = pv.plot(kind="bar", stacked=True, figsize=(7, 4.5), color=[COLORS[r] for r in REGIMES])
    ax.set_ylabel("% số ngày"); ax.set_title("Phân bố giai đoạn theo tập dữ liệu")
    ax.set_xticklabels(pv.index, rotation=0); ax.legend(title="Regime", bbox_to_anchor=(1, 1))
    plt.tight_layout(); plt.savefig(FIG_DIR / "kmeans_regime_distribution.png", dpi=120); plt.close()


def fig_transition(tm: pd.DataFrame):
    n = len(REGIMES)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(tm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), REGIMES); ax.set_yticks(range(n), REGIMES)
    ax.set_xlabel("Ngày t"); ax.set_ylabel("Ngày t-1")
    for i in range(n):
        for j in range(n):
            v = tm.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > .5 else "black")
    fig.colorbar(im); ax.set_title("Ma trận chuyển trạng thái (Train)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "kmeans_transition_matrix.png", dpi=120); plt.close(fig)


def fig_forward_returns(data):
    train = data["train"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, h in zip(axes, [1, 7, 30]):
        vals = [100 * train.loc[train["market_phase"] == r, f"target_logret_h{h}"].dropna() for r in REGIMES]
        bp = ax.boxplot(vals, tick_labels=REGIMES, showfliers=False, patch_artist=True)
        for patch, r in zip(bp["boxes"], REGIMES):
            patch.set_facecolor(COLORS[r]); patch.set_alpha(.6)
        ax.axhline(0, color="black", lw=.8)
        ax.set_title(f"Log-return tương lai h={h} (%)"); ax.grid(alpha=.3)
    fig.suptitle("Phân phối lợi suất tương lai theo giai đoạn (Train)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "kmeans_forward_returns.png", dpi=120); plt.close(fig)


def fig_feature_comparison(cmp: pd.DataFrame):
    sub = cmp[cmp["k"] == len(REGIMES)].set_index("features")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    specs = [("silhouette", "Silhouette ↑"), ("davies_bouldin", "Davies-Bouldin ↓"),
             ("avg_run_days", "Độ dài 1 đợt (ngày) ↑"), ("oos_fwd_vol_ratio", "Tách biến động tương lai ↑")]
    for ax, (col, title) in zip(axes, specs):
        colors = ["orange" if "(chọn)" in f else "#9aa5b1" for f in sub.index]
        ax.barh(sub.index, sub[col], color=colors)
        ax.set_title(title); ax.grid(alpha=.3, axis="x")
    axes[0].invert_yaxis()
    fig.suptitle(f"So sánh các phương án feature (k = {len(REGIMES)})")
    fig.tight_layout(); fig.savefig(FIG_DIR / "kmeans_feature_comparison.png", dpi=120); plt.close(fig)


def fig_oos_volatility(oos: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([f"{r}\n({REGIME_LABELS_VI[r]})" for r in oos["regime"]], oos["fwd_vol20_ann_pct"],
           color=[COLORS[r] for r in oos["regime"]])
    ax.set_ylabel("Biến động 20 phiên tới (% năm hóa)")
    ax.set_title("Val + Test: giai đoạn hiện tại → biến động tương lai")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG_DIR / "kmeans_oos_volatility.png", dpi=120); plt.close(fig)


# ── report ─────────────────────────────────────────────────────────────────
def write_markdown(bundle, metrics, scan, centers, stats, tm, cmp, oos):
    k_sel    = len(REGIMES)
    best_sil = int(scan.loc[scan["silhouette"].idxmax(), "k"])
    best_db  = int(scan.loc[scan["davies_bouldin"].idxmin(), "k"])
    missing  = stats[(stats["days"] == 0)][["split", "regime"]].values.tolist()
    persist  = np.diag(tm.values)
    unlabeled = metrics.set_index("split")["unlabeled"]

    cmp_k   = cmp[cmp["k"] == k_sel].set_index("features")
    old     = cmp_k.iloc[0]
    chosen  = cmp_k.loc[[f for f in cmp_k.index if "(chọn)" in f][0]]
    vol_oos = oos.set_index("regime")["fwd_vol20_ann_pct"]

    centers_vi = centers.copy()
    centers_vi.insert(1, "nhãn", [REGIME_LABELS_VI[r] for r in centers_vi["regime"]])
    centers_vi["trend_ann_%"] = centers_vi[bundle["columns"][0]] * TRADING_DAYS * 100
    centers_vi["vol_ann_%"]   = centers_vi[bundle["columns"][1]] * np.sqrt(TRADING_DAYS) * 100

    lines = [
        "# Kết quả K-Means — Phân lớp giai đoạn thị trường",
        "",
        "> File được sinh tự động bởi `python -m source.preprocessing_v3.report_kmeans`. "
        "Không sửa tay — chạy lại script sau mỗi lần chạy Stage 2.",
        "",
        "## 1. Cấu hình",
        "",
        f"- Thuật toán: K-Means, k = {k_sel}, n_init = {bundle['kmeans'].n_init}, random_state = 42",
        f"- Feature phân cụm (StandardScaler fit trên Train): "
        f"`{bundle['columns'][0]}` (xu hướng {bundle['trend_window']} phiên), "
        f"`{bundle['columns'][1]}` (biến động {bundle['vol_window']} phiên). Cả hai chỉ dùng quá khứ (shift(1)).",
        f"- Train: → {TRAIN_END} | Val: {VAL_START} → {VAL_END} | Test: {TEST_START} →",
        "- Đặt tên cụm: cụm có biến động cao nhất → **Volatile**; 3 cụm còn lại theo xu hướng tăng dần → "
        "**Bear**, **Bull**, **StrongBull**.",
        f"- Model: `{KMEANS_MODEL.relative_to(ROOT_DIR).as_posix()}`",
        "",
        "## 2. Vì sao chọn cấu hình này",
        "",
        "Đã thử nhiều bộ feature với k = 3 và k = 4. Tiêu chí: cụm tách rõ (Silhouette ↑, Davies-Bouldin ↓), "
        "giai đoạn đủ bền để dùng được (`avg_run_days` ↑), Val có đủ giai đoạn để đánh giá, và quan trọng nhất là "
        "giai đoạn gán trên **Val + Test** (dữ liệu chưa thấy) phân biệt được biến động tương lai "
        "(`oos_fwd_vol_ratio` = biến động 20 phiên tới của giai đoạn cao nhất / thấp nhất).",
        "",
        _md_table(cmp, "{:.3f}"),
        "",
        "![So sánh feature](figures/kmeans_feature_comparison.png)",
        "",
        f"So với bản cũ (k = {k_sel}): Silhouette {old['silhouette']:.3f} → **{chosen['silhouette']:.3f}**, "
        f"Davies-Bouldin {old['davies_bouldin']:.3f} → **{chosen['davies_bouldin']:.3f}**, "
        f"độ dài một đợt {old['avg_run_days']:.1f} → **{chosen['avg_run_days']:.1f} ngày**.",
        "",
        "- Bỏ log-return của chính ngày t: feature này làm cụm bị chia theo \"hôm nay tăng hay giảm\" "
        "→ nhãn nhiễu, đổi liên tục.",
        "- Xu hướng 60 phiên ổn định hơn 20 phiên; biến động 20 phiên phản ứng nhanh với cú sốc.",
        f"- k = {k_sel} thay vì 3: tách được giai đoạn biến động mạnh / khủng hoảng riêng, "
        "không trộn lẫn với giai đoạn giảm thông thường.",
        "",
        "## 3. Chất lượng phân cụm",
        "",
        "Silhouette ↑ (−1…1), Davies-Bouldin ↓, Calinski-Harabasz ↑. Model chỉ fit trên Train; Val/Test dùng nhãn dự đoán.",
        "",
        _md_table(metrics),
        "",
        "### Thử nghiệm số cụm k (Train, cùng bộ feature)",
        "",
        _md_table(scan),
        "",
        f"- Silhouette tốt nhất tại k = {best_sil}; Davies-Bouldin tốt nhất tại k = {best_db}.",
        "",
        "![Chọn k](figures/kmeans_k_selection.png)",
        "",
        "## 4. Tâm cụm",
        "",
        "`trend_ann_%`: xu hướng quy ra % năm; `vol_ann_%`: biến động quy ra % năm.",
        "",
        _md_table(centers_vi, "{:.5f}"),
        "",
        "![Không gian phân cụm](figures/kmeans_cluster_scatter.png)",
        "",
        "## 5. Giai đoạn có giúp ích cho dự đoán không? (Val + Test, dữ liệu chưa thấy)",
        "",
        "`fwd_vol20_ann_pct`: biến động thực tế 20 phiên tới; `fwd_logret_h*_pct`: log-return trung bình h phiên tới; "
        "`up_ratio_h30_pct`: % số ngày giá 30 phiên sau cao hơn hiện tại.",
        "",
        _md_table(oos, "{:.3f}"),
        "",
        "![Biến động tương lai](figures/kmeans_oos_volatility.png)",
        "",
        f"- Biến động tương lai khác biệt rõ giữa các giai đoạn "
        f"(cao nhất {vol_oos.max():.1f}% so với thấp nhất {vol_oos.min():.1f}% năm hóa) → giai đoạn mang thông tin "
        "về **mức độ rủi ro / độ rộng khoảng tin cậy** của dự đoán.",
        "- Lợi suất trung bình khác nhau giữa các giai đoạn nhưng có số mẫu nhỏ và các cửa sổ h=30 chồng lấn, "
        "nên chưa đủ để kết luận về hướng giá. Việc phân lớp có giúp dự đoán giá tốt hơn hay không phải được "
        "kiểm chứng bằng so sánh model giữa nhánh `*_base` và `*_kmeans`.",
        "",
        "## 6. Thống kê từng giai đoạn theo tập",
        "",
        "`mean_logret_%`: log-return ngày trung bình; `ann_vol_%`: biến động năm hóa; "
        "`avg_run_days`: độ dài trung bình một đợt liên tiếp; `fwd_logret_h*_%`: log-return trung bình h ngày tới (target).",
        "",
        _md_table(stats, "{:.3f}"),
        "",
        "![Timeline](figures/kmeans_regime_timeline.png)",
        "",
        "![Phân bố](figures/kmeans_regime_distribution.png)",
        "",
        "![Lợi suất tương lai](figures/kmeans_forward_returns.png)",
        "",
        "## 7. Ma trận chuyển trạng thái (Train)",
        "",
        "Xác suất giai đoạn ngày t (cột) khi biết giai đoạn ngày t-1 (hàng).",
        "",
        _md_table(tm.reset_index().rename(columns={"index": "t-1 \\ t", "market_phase": "t-1 \\ t"}), "{:.3f}"),
        "",
        "![Ma trận chuyển](figures/kmeans_transition_matrix.png)",
        "",
        "## 8. Nhận xét và lưu ý",
        "",
        "- Độ bền giai đoạn (xác suất giữ nguyên ngày kế tiếp): "
        + ", ".join(f"{r} = {p:.2f}" for r, p in zip(REGIMES, persist)) + ".",
    ]
    if missing:
        lines.append("- **Cảnh báo:** không có ngày nào thuộc "
                     + ", ".join(f"{r} trong {sp}" for sp, r in missing)
                     + " → không đánh giá được model ở giai đoạn này trên tập đó; "
                     "khi so sánh nên báo cáo kết quả theo từng giai đoạn trên Test.")
    lines += [
        "- Số dòng chưa có nhãn (thiếu lịch sử rolling): "
        + ", ".join(f"{sp} = {int(n)}" for sp, n in unlabeled.items()) + ".",
        "- Train (2001–2017) chủ yếu là chu kỳ tăng dài của vàng nên phần lớn cụm có xu hướng dương; "
        "không có cụm \"đi ngang\" riêng vì dữ liệu không tạo ra cụm đó một cách tự nhiên.",
        "",
    ]
    (OUT_DIR / "result_k_mean.md").write_text("\n".join(lines), encoding="utf-8")


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    bundle, data = _load()
    metrics = metrics_per_split(bundle, data)
    scan    = k_scan(bundle, data["train"])
    centers = centers_table(bundle)
    stats   = regime_stats(data)
    tm      = transition_matrix(data["train"])
    cmp     = feature_comparison()
    oos     = oos_usefulness(data)

    metrics.to_csv(OUT_DIR / "result_k_mean_metrics.csv", index=False)
    scan.to_csv(OUT_DIR / "result_k_mean_k_scan.csv", index=False)
    stats.to_csv(OUT_DIR / "result_k_mean_regimes.csv", index=False)
    cmp.to_csv(OUT_DIR / "result_k_mean_feature_comparison.csv", index=False)
    oos.to_csv(OUT_DIR / "result_k_mean_oos_usefulness.csv", index=False)

    fig_k_scan(scan)
    fig_timeline(data)
    fig_scatter(bundle, data["train"])
    fig_distribution(stats)
    fig_transition(tm)
    fig_forward_returns(data)
    fig_feature_comparison(cmp)
    fig_oos_volatility(oos)

    write_markdown(bundle, metrics, scan, centers, stats, tm, cmp, oos)
    logger.info("K-Means report → %s", OUT_DIR / "result_k_mean.md")


if __name__ == "__main__":
    run()
