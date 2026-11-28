"""
Báo cáo so sánh: dự đoán CÓ phân lớp K-Means vs KHÔNG phân lớp (cho mọi model đã train).

Chạy từ thư mục gốc dự án (DTMining/), sau khi train model:
    python -m source.model.report_forecast

Đọc   : results/forecasting/<model>/{metrics,predictions,feature_importance}.csv
Output: results/forecasting/result_forecast.md
        results/forecasting/result_forecast_comparison.csv
        results/forecasting/result_forecast_by_regime.csv
        results/forecasting/figures/*.png
"""
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source.model.common.data import REGIME_FEATURES
from source.model.common.intervals import LEVELS, apply_widths, fit_widths, interval_scores, winkler
from source.model.common.metrics import diebold_mariano, dm_on_losses, metrics_by_group
from source.preprocessing_v3.config import BRANCHES, FORECAST_RESULTS, HORIZONS, ROOT_DIR
from source.preprocessing_v3.step09_clustering import REGIME_LABELS_VI, REGIMES

logger = logging.getLogger(__name__)

FIG_DIR = FORECAST_RESULTS / "figures"
COLORS  = {"naive": "#9aa5b1", "base": "#1f77b4", "kmeans": "#ff7f0e"}
LABELS  = {"naive": "Naive (giá t+h = giá t)", **BRANCHES}
ALPHA   = 0.05


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep  = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _load():
    metrics, preds, imps = [], [], []
    for d in sorted(p for p in FORECAST_RESULTS.iterdir() if p.is_dir() and (p / "metrics.csv").exists()):
        metrics.append(pd.read_csv(d / "metrics.csv"))
        preds.append(pd.read_csv(d / "predictions.csv", parse_dates=["Date"]))
        if (d / "feature_importance.csv").exists():
            imps.append(pd.read_csv(d / "feature_importance.csv"))
    if not metrics:
        raise FileNotFoundError(f"Chưa có kết quả model nào trong {FORECAST_RESULTS}")
    return (pd.concat(metrics, ignore_index=True), pd.concat(preds, ignore_index=True),
            pd.concat(imps, ignore_index=True) if imps else pd.DataFrame())


# ── phân tích ──────────────────────────────────────────────────────────────
def comparison(metrics: pd.DataFrame, preds: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for (model, h), g in preds[preds["split"] == split].groupby(["model", "horizon"]):
        b = g[g["branch"] == "base"].set_index("Date")
        k = g[g["branch"] == "kmeans"].set_index("Date")
        idx = b.index.intersection(k.index)
        e_b = (b.loc[idx, "y_true"] - b.loc[idx, "y_pred"]).to_numpy()
        e_k = (k.loc[idx, "y_true"] - k.loc[idx, "y_pred"]).to_numpy()
        stat, p = diebold_mariano(e_k, e_b, int(h))
        m = metrics[(metrics["model"] == model) & (metrics["horizon"] == h) & (metrics["split"] == split)]
        mb, mk = m[m["branch"] == "base"].iloc[0], m[m["branch"] == "kmeans"].iloc[0]
        delta = 100 * (mk["MAE"] - mb["MAE"]) / mb["MAE"]
        if p < ALPHA:
            verdict = "K-Means TỐT HƠN" if stat < 0 else "K-Means KÉM HƠN"
        else:
            verdict = "Không khác biệt có ý nghĩa"
        rows.append({"model": model, "horizon": int(h),
                     "MAE_base": mb["MAE"], "MAE_kmeans": mk["MAE"], "ΔMAE_%": delta,
                     "MASE_base": mb["MASE"], "MASE_kmeans": mk["MASE"],
                     "DA_base": mb["DA"], "DA_kmeans": mk["DA"],
                     "DM_stat": stat, "p_value": p, "kết_luận": verdict})
    return pd.DataFrame(rows)


def by_regime(preds: pd.DataFrame) -> pd.DataFrame:
    test = preds[(preds["split"] == "test") & preds["branch"].isin(["base", "kmeans"])]
    rows = []
    for (model, h, branch), g in test.groupby(["model", "horizon", "branch"]):
        t = metrics_by_group(g, "regime_label").assign(model=model, horizon=h, branch=branch)
        rows.append(t)
    df = pd.concat(rows)
    wide = df.pivot_table(index=["model", "horizon", "regime_label"], columns="branch",
                          values=["MAE", "n"]).reset_index()
    wide.columns = ["model", "horizon", "regime", "MAE_base", "MAE_kmeans", "n", "_n2"]
    wide = wide.drop(columns="_n2")
    wide["ΔMAE_%"] = 100 * (wide["MAE_kmeans"] - wide["MAE_base"]) / wide["MAE_base"]
    wide["nhãn"] = wide["regime"].map(REGIME_LABELS_VI)
    wide["regime"] = pd.Categorical(wide["regime"], REGIMES, ordered=True)
    wide = wide.sort_values(["model", "horizon", "regime"])
    return wide[["model", "horizon", "regime", "nhãn", "n", "MAE_base", "MAE_kmeans", "ΔMAE_%"]]


def interval_analysis(preds: pd.DataFrame):
    """Khoảng dự đoán trên Test: base (độ rộng chung) vs kmeans (độ rộng theo giai đoạn)."""
    summary, per_regime, curves = [], [], {}
    for (model, h), g in preds.groupby(["model", "horizon"]):
        winklers = {}
        for branch in ("base", "kmeans"):
            gb   = g[g["branch"] == branch]
            cal  = gb[gb["split"] == "cv"]
            test = gb[gb["split"] == "test"].sort_values("Date").reset_index(drop=True)
            widths = fit_widths(cal, conditional=(branch == "kmeans"))
            for lv in LEVELS:
                lo, hi = apply_widths(test, widths, lv)
                summary.append({"model": model, "horizon": int(h), "level_%": int(lv * 100),
                                "branch": branch, "n_cal": len(cal),
                                **interval_scores(test["y_true"], lo, hi, lv)})
                winklers[(branch, lv)] = winkler(test["y_true"], lo, hi, lv)
                curves[(model, int(h), branch, lv)] = (test, lo, hi)
                for r, idx in test.groupby("regime_label").groups.items():
                    per_regime.append({"model": model, "horizon": int(h), "level_%": int(lv * 100),
                                       "branch": branch, "regime": r, "n": len(idx),
                                       **interval_scores(test.loc[idx, "y_true"], lo[idx], hi[idx], lv)})
        for lv in LEVELS:
            stat, p = dm_on_losses(winklers[("kmeans", lv)], winklers[("base", lv)], int(h))
            for row in summary:
                if (row["model"], row["horizon"], row["level_%"]) == (model, int(h), int(lv * 100)):
                    row["DM_winkler"], row["p_value"] = stat, p

    summ = pd.DataFrame(summary)
    wide = summ.pivot_table(index=["model", "horizon", "level_%"], columns="branch",
                            values=["coverage_%", "width_usd", "winkler"]).reset_index()
    wide.columns = [f"{a}_{b}" if b else a for a, b in wide.columns]
    tests = summ.drop_duplicates(["model", "horizon", "level_%"])[["model", "horizon", "level_%", "DM_winkler", "p_value"]]
    wide = wide.merge(tests, on=["model", "horizon", "level_%"])
    wide["Δwinkler_%"] = 100 * (wide["winkler_kmeans"] - wide["winkler_base"]) / wide["winkler_base"]
    wide["kết_luận"] = np.where(wide["p_value"] < ALPHA,
                                np.where(wide["DM_winkler"] < 0, "K-Means TỐT HƠN", "K-Means KÉM HƠN"),
                                "Không khác biệt có ý nghĩa")
    cols = ["model", "horizon", "level_%", "coverage_%_base", "coverage_%_kmeans", "width_usd_base",
            "width_usd_kmeans", "winkler_base", "winkler_kmeans", "Δwinkler_%", "DM_winkler", "p_value", "kết_luận"]

    reg = pd.DataFrame(per_regime)
    reg_w = reg.pivot_table(index=["model", "horizon", "level_%", "regime", "n"], columns="branch",
                            values=["coverage_%", "width_usd"]).reset_index()
    reg_w.columns = [f"{a}_{b}" if b else a for a, b in reg_w.columns]
    reg_w["regime"] = pd.Categorical(reg_w["regime"], REGIMES, ordered=True)
    reg_w = reg_w.sort_values(["model", "horizon", "level_%", "regime"])
    return wide[cols], reg_w, curves


def regime_importance(imps: pd.DataFrame) -> pd.DataFrame:
    if imps.empty:
        return pd.DataFrame()
    rows = []
    for (model, h), g in imps[imps["branch"] == "kmeans"].groupby(["model", "horizon"]):
        g = g.sort_values("importance_gain", ascending=False).reset_index(drop=True)
        g["rank"] = np.arange(1, len(g) + 1)
        share = g["importance_gain"] / g["importance_gain"].sum() * 100
        reg = g[g["feature"].isin(REGIME_FEATURES)]
        rows.append({"model": model, "horizon": h, "n_features": len(g),
                     "tổng_gain_regime_%": float(share[reg.index].sum()),
                     "rank_tốt_nhất": int(reg["rank"].min()) if len(reg) else np.nan,
                     "feature_regime_mạnh_nhất": reg.iloc[0]["feature"] if len(reg) else ""})
    return pd.DataFrame(rows)


# ── hình ───────────────────────────────────────────────────────────────────
def fig_mase(metrics: pd.DataFrame, model: str):
    test = metrics[(metrics["split"] == "test") & metrics["model"].isin([model, "naive"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.27
    for i, br in enumerate(["naive", "base", "kmeans"]):
        vals = [test[(test["branch"] == br) & (test["horizon"] == h)]["MASE"].iloc[0] for h in HORIZONS]
        bars = ax.bar(np.arange(len(HORIZONS)) + (i - 1) * w, vals, w, color=COLORS[br], label=LABELS[br])
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.axhline(1, color="black", ls="--", lw=.8)
    ax.set_xticks(range(len(HORIZONS)), [f"h = {h}" for h in HORIZONS])
    ax.set_ylabel("MASE (< 1 là tốt hơn naive)"); ax.set_title(f"{model.upper()} — MASE trên Test (2020–2022)")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{model}_mase_by_horizon.png", dpi=120); plt.close(fig)


def fig_predictions(preds: pd.DataFrame, model: str):
    test = preds[(preds["split"] == "test") & (preds["model"] == model)]
    fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(15, 3.6 * len(HORIZONS)), sharex=True)
    for ax, h in zip(axes, HORIZONS):
        g = test[test["horizon"] == h]
        b = g[g["branch"] == "base"].sort_values("Date")
        k = g[g["branch"] == "kmeans"].sort_values("Date")
        tgt_date = b["Date"]  # trục x: ngày dự đoán được đưa ra
        ax.plot(tgt_date, b["y_true"], color="black", lw=1, label=f"Giá thực tế t+{h}")
        ax.plot(tgt_date, b["y_pred"], color=COLORS["base"], lw=.9, alpha=.8, label=LABELS["base"])
        ax.plot(k["Date"], k["y_pred"], color=COLORS["kmeans"], lw=.9, alpha=.8, label=LABELS["kmeans"])
        ax.set_title(f"h = {h}"); ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{model.upper()} — Dự đoán trên Test (trục x: ngày đưa ra dự đoán)")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{model}_test_predictions.png", dpi=110); plt.close(fig)


def fig_cum_error(preds: pd.DataFrame, model: str):
    test = preds[(preds["split"] == "test") & (preds["model"] == model)]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for h, ls in zip(HORIZONS, ["-", "--", ":"]):
        g = test[test["horizon"] == h]
        b = g[g["branch"] == "base"].set_index("Date").sort_index()
        k = g[g["branch"] == "kmeans"].set_index("Date").sort_index()
        d = ((b["y_true"] - b["y_pred"]).abs() - (k["y_true"] - k["y_pred"]).abs()).cumsum()
        ax.plot(d.index, d.values, ls=ls, label=f"h = {h}")
    ax.axhline(0, color="black", lw=.8)
    ax.set_ylabel("Σ (|e_base| − |e_kmeans|)  (USD)")
    ax.set_title(f"{model.upper()} — Lợi thế tích lũy của nhánh K-Means trên Test (đi lên = K-Means tốt hơn)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{model}_cumulative_advantage.png", dpi=120); plt.close(fig)


def fig_regime(reg: pd.DataFrame, model: str):
    sub = reg[reg["model"] == model]
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(16, 4.5))
    for ax, h in zip(axes, HORIZONS):
        g = sub[sub["horizon"] == h]
        x = np.arange(len(g))
        ax.bar(x - .2, g["MAE_base"], .4, color=COLORS["base"], label=LABELS["base"])
        ax.bar(x + .2, g["MAE_kmeans"], .4, color=COLORS["kmeans"], label=LABELS["kmeans"])
        ax.set_xticks(x, [f"{r}\n(n={int(n)})" for r, n in zip(g["regime"], g["n"])], fontsize=8)
        ax.set_title(f"h = {h}"); ax.set_ylabel("MAE (USD)"); ax.grid(alpha=.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{model.upper()} — MAE trên Test theo giai đoạn thị trường")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{model}_mae_by_regime.png", dpi=120); plt.close(fig)


def fig_intervals(curves: dict, model: str, horizon: int = 7, level: float = 0.80):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, sharey=True)
    for ax, branch in zip(axes, ("base", "kmeans")):
        test, lo, hi = curves[(model, horizon, branch, level)]
        inside = (test["y_true"] >= lo) & (test["y_true"] <= hi)
        ax.fill_between(test["Date"], lo, hi, color=COLORS[branch], alpha=.3,
                        label=f"Khoảng {int(level * 100)}%")
        ax.plot(test["Date"], test["y_true"], color="black", lw=.8, label=f"Giá thực tế t+{horizon}")
        ax.scatter(test["Date"][~inside], test["y_true"][~inside], s=6, color="red", zorder=3,
                   label="Nằm ngoài khoảng")
        ax.set_title(f"{LABELS[branch]} — coverage {inside.mean() * 100:.1f}%, "
                     f"độ rộng TB {np.mean(hi - lo):.1f} USD")
        ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{model.upper()} — Khoảng dự đoán {int(level * 100)}% trên Test, h = {horizon}")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{model}_intervals_h{horizon}.png", dpi=110); plt.close(fig)


def fig_interval_regime(reg_int: pd.DataFrame, model: str, level: int = 80):
    sub = reg_int[(reg_int["model"] == model) & (reg_int["level_%"] == level)]
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(16, 4.5), sharey=True)
    for ax, h in zip(axes, HORIZONS):
        g = sub[sub["horizon"] == h]
        x = np.arange(len(g))
        ax.bar(x - .2, g["coverage_%_base"], .4, color=COLORS["base"], label=LABELS["base"])
        ax.bar(x + .2, g["coverage_%_kmeans"], .4, color=COLORS["kmeans"], label=LABELS["kmeans"])
        ax.axhline(level, color="black", ls="--", lw=1, label=f"Mức danh nghĩa {level}%")
        ax.set_xticks(x, [f"{r}\n(n={int(n)})" for r, n in zip(g["regime"], g["n"])], fontsize=8)
        ax.set_title(f"h = {h}"); ax.grid(alpha=.3, axis="y")
    axes[0].set_ylabel("Coverage (%)"); axes[0].legend(fontsize=8)
    fig.suptitle(f"{model.upper()} — Coverage khoảng {level}% theo giai đoạn (Test): càng sát đường đứt càng tốt")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{model}_interval_coverage_by_regime.png", dpi=120); plt.close(fig)


# ── báo cáo ────────────────────────────────────────────────────────────────
def write_markdown(metrics, cmp_test, cmp_val, reg, reg_imp, int_tbl, int_reg, models):
    disp = lambda df: df.assign(branch=df["branch"].map(LABELS))[
        ["model", "branch", "horizon", "MAE", "RMSE", "MAPE", "R2", "MASE", "DA", "n"]]
    test_tbl = disp(metrics[metrics["split"] == "test"].sort_values(["horizon", "model", "branch"]))
    val_tbl  = disp(metrics[metrics["split"] == "val"].sort_values(["horizon", "model", "branch"]))

    n_better = int((cmp_test["kết_luận"] == "K-Means TỐT HƠN").sum())
    n_worse  = int((cmp_test["kết_luận"] == "K-Means KÉM HƠN").sum())
    n_same   = len(cmp_test) - n_better - n_worse
    avg_delta = cmp_test["ΔMAE_%"].mean()
    naive_test = metrics[(metrics["split"] == "test") & (metrics["branch"] != "naive")]
    beats_naive = int((naive_test["MASE"] < 1).sum())
    i_better = int((int_tbl["kết_luận"] == "K-Means TỐT HƠN").sum())
    i_worse  = int((int_tbl["kết_luận"] == "K-Means KÉM HƠN").sum())
    i_delta  = int_tbl["Δwinkler_%"].mean()
    vol_rows = int_reg[(int_reg["regime"] == "Volatile")]
    cal_rows = int_reg[(int_reg["regime"] == "Bull")]

    lines = [
        "# Kết quả dự đoán — Có phân lớp K-Means vs Không phân lớp",
        "",
        "> Sinh tự động bởi `python -m source.model.report_forecast`. Không sửa tay.",
        "",
        "## 1. Thiết kế thí nghiệm",
        "",
        "- **Câu hỏi:** phân lớp giai đoạn thị trường bằng K-Means trước khi dự đoán có giúp dự đoán giá vàng tốt hơn không?",
        "- **Nhánh không phân lớp (base):** dữ liệu `*_base.csv` → feature riêng của từng model.",
        "- **Nhánh có phân lớp (kmeans):** dữ liệu `*_kmeans.csv` → cùng feature đó + nhãn giai đoạn do `kmeans.joblib` gán "
        f"(`{'`, `'.join(REGIME_FEATURES)}`).",
        "- Trong cùng một model, hai nhánh dùng **cùng các dòng dữ liệu, cùng feature gốc, cùng target, cùng quy trình tuning** "
        "→ khác biệt chỉ đến từ thông tin phân lớp.",
        "- Target: log-return h phiên tới; giá dự đoán = giá hôm nay × exp(ŷ). Đánh giá trên giá (USD/oz).",
        "- Tuning: Optuna trên walk-forward folds cuối của Train (có purge theo ngày target). "
        "Validation: fit Train + early stopping. Test: refit Train+Val, đánh giá trên 2020–2022.",
        "- **Naive baseline:** giá t+h = giá t. **MASE < 1** nghĩa là tốt hơn naive.",
        "- **Diebold-Mariano (DM):** kiểm định sai số tuyệt đối của 2 nhánh có khác nhau thật không "
        f"(DM < 0: K-Means sai số nhỏ hơn; có ý nghĩa nếu p < {ALPHA}).",
        "",
        "## 2. Tóm tắt kết luận (Test 2020–2022)",
        "",
        f"1. **Dự đoán điểm (giá):** K-Means tốt hơn có ý nghĩa ở {n_better}/{len(cmp_test)} khoảng dự đoán, "
        f"kém hơn ở {n_worse}, không khác biệt ở {n_same}; MAE thay đổi trung bình {avg_delta:+.2f}%. "
        f"{beats_naive}/{len(naive_test)} cấu hình tốt hơn naive.",
        f"2. **Khoảng dự đoán (độ tin cậy):** K-Means tốt hơn có ý nghĩa ở {i_better}/{len(int_tbl)} cấu hình "
        f"(h × mức tin cậy), kém hơn ở {i_worse}; Winkler score thay đổi trung bình {i_delta:+.2f}% (âm = tốt hơn).",
        "",
        "## 3. Dự đoán điểm — có phân lớp vs không phân lớp",
        "",
        _md_table(cmp_test, "{:.4f}"),
        "",
    ]
    for m in models:
        lines += [f"![MASE {m}](figures/{m}_mase_by_horizon.png)", "",
                  f"![Lợi thế tích lũy {m}](figures/{m}_cumulative_advantage.png)", ""]

    lines += [
        "## 4. Khoảng dự đoán — có phân lớp vs không phân lớp",
        "",
        "Hiệu chỉnh trên sai số out-of-fold walk-forward 2004–2017 (đủ mọi giai đoạn), đánh giá trên Test. "
        "Nhánh base dùng một độ rộng chung; nhánh kmeans dùng độ rộng theo giai đoạn do K-Means gán. "
        "**Coverage** càng sát mức danh nghĩa càng tốt; **Winkler** (độ rộng + phạt khi giá thực nằm ngoài) càng nhỏ càng tốt.",
        "",
        _md_table(int_tbl, "{:.3f}"),
        "",
        "### Coverage và độ rộng theo giai đoạn",
        "",
        _md_table(int_reg, "{:.2f}"),
        "",
    ]
    if len(vol_rows) and len(cal_rows):
        v80 = vol_rows[vol_rows["level_%"] == 80]
        c80 = cal_rows[cal_rows["level_%"] == 80]
        lines += [
            f"- Giai đoạn **Volatile** (khoảng 80%): coverage base trung bình {v80['coverage_%_base'].mean():.1f}% "
            f"→ kmeans {v80['coverage_%_kmeans'].mean():.1f}%; độ rộng {v80['width_usd_base'].mean():.1f} → "
            f"{v80['width_usd_kmeans'].mean():.1f} USD.",
            f"- Giai đoạn **Bull** (khoảng 80%): coverage base {c80['coverage_%_base'].mean():.1f}% → "
            f"kmeans {c80['coverage_%_kmeans'].mean():.1f}%; độ rộng {c80['width_usd_base'].mean():.1f} → "
            f"{c80['width_usd_kmeans'].mean():.1f} USD.",
            "",
        ]
    for m in models:
        lines += [f"![Khoảng dự đoán {m}](figures/{m}_intervals_h7.png)", "",
                  f"![Coverage theo giai đoạn {m}](figures/{m}_interval_coverage_by_regime.png)", ""]

    lines += [
        "## 5. Chi tiết chỉ số dự đoán điểm trên Test",
        "",
        "MAE, RMSE: USD/oz · MAPE: % · DA: % dự đoán đúng chiều tăng/giảm.",
        "",
        _md_table(test_tbl),
        "",
    ]
    for m in models:
        lines += [f"![Dự đoán {m}](figures/{m}_test_predictions.png)", ""]

    lines += [
        "## 6. Dự đoán điểm theo từng giai đoạn thị trường (Test)",
        "",
        "Cho biết phân lớp giúp/hại ở giai đoạn nào.",
        "",
        _md_table(reg, "{:.3f}"),
        "",
    ]
    for m in models:
        lines += [f"![Theo giai đoạn {m}](figures/{m}_mae_by_regime.png)", ""]

    if not reg_imp.empty:
        lines += [
            "## 7. Model có dùng thông tin phân lớp không? (Feature importance, nhánh kmeans)",
            "",
            _md_table(reg_imp, "{:.2f}"),
            "",
        ]

    lines += [
        "## 8. Dự đoán điểm trên Validation (2018–2019)",
        "",
        _md_table(cmp_val, "{:.4f}"),
        "",
        _md_table(val_tbl),
        "",
        "## 9. Ghi chú khi đọc kết quả",
        "",
        "- R² trên giá luôn cao vì giá hôm nay đã giải thích gần hết giá ngày mai; hãy so MASE và DA với naive.",
        "- Dự đoán giá vàng ngắn hạn gần như bước ngẫu nhiên: MASE quanh 1 và DA quanh 50% là bình thường. "
        "Điểm cần đọc là **chênh lệch giữa 2 nhánh** và **kiểm định DM**.",
        "- Val không có giai đoạn Volatile (xem `results/kmeans/result_k_mean.md`), nên Test là căn cứ chính.",
        "",
    ]
    (FORECAST_RESULTS / "result_forecast.md").write_text("\n".join(lines), encoding="utf-8")


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics, preds, imps = _load()
    models = sorted(preds["model"].unique())

    cmp_test = comparison(metrics, preds, "test")
    cmp_val  = comparison(metrics, preds, "val")
    reg      = by_regime(preds)
    reg_imp  = regime_importance(imps)
    int_tbl, int_reg, curves = interval_analysis(preds)

    cmp_test.to_csv(FORECAST_RESULTS / "result_forecast_comparison.csv", index=False)
    reg.to_csv(FORECAST_RESULTS / "result_forecast_by_regime.csv", index=False)
    int_tbl.to_csv(FORECAST_RESULTS / "result_forecast_intervals.csv", index=False)
    int_reg.to_csv(FORECAST_RESULTS / "result_forecast_intervals_by_regime.csv", index=False)

    for m in models:
        fig_mase(metrics, m)
        fig_predictions(preds, m)
        fig_cum_error(preds, m)
        fig_regime(reg, m)
        fig_intervals(curves, m)
        fig_interval_regime(int_reg, m)

    write_markdown(metrics, cmp_test, cmp_val, reg, reg_imp, int_tbl, int_reg, models)
    logger.info("Forecast report → %s", FORECAST_RESULTS / "result_forecast.md")


if __name__ == "__main__":
    run()
