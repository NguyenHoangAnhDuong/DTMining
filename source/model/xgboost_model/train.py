"""
XGBoost — Dự đoán giá vàng cho 2 nhánh (base / kmeans) × 3 khoảng dự đoán (h = 1, 7, 30).

Chạy từ thư mục gốc dự án (DTMining/), sau Stage 1 + Stage 2:
    python -m source.model.xgboost_model.train

Quy trình cho mỗi (nhánh, h) — giống hệt nhau giữa 2 nhánh:
  1. Dữ liệu chung (common/data.py) + feature riêng của XGBoost (features.py), target = log-return h phiên
  2. Optuna TPE tìm hyperparameter trên các fold walk-forward cuối của Train (có purge)
  3. Model Val : fit Train, early-stopping theo Val → số liệu Validation + số cây tốt nhất
  4. Model Test: fit lại Train+Val với số cây tốt nhất → số liệu Test (dữ liệu chưa từng thấy)
  5. Quy đổi log-return → giá, đánh giá MAE, RMSE, MAPE, R², MASE, DA; so với naive

Output (theo docs/result_contract.md): results/forecasting/xgboost/
  metrics.csv, predictions.csv, feature_importance.csv, metadata.json, models/*.json

Web app: dùng `predict_latest(history, branch, horizon)`.
"""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from source.model.common.data import REGIME_FEATURES, prediction_frame, purge_fold, to_price
from source.model.common.intervals import fit_widths
from source.model.common.metrics import forecast_metrics, metrics_table
from source.model.xgboost_model.features import build_features, make_dataset
from source.preprocessing_v3.config import BRANCHES, FORECAST_RESULTS, HORIZONS, SPLIT_DIR
from source.preprocessing_v3.step09_clustering import assign_regime

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

MODEL_NAME   = "xgboost"
OUT_DIR      = FORECAST_RESULTS / MODEL_NAME
MODEL_DIR    = OUT_DIR / "models"
N_TRIALS     = 30
TUNE_FOLDS   = 3
EARLY_STOP   = 50
MAX_TREES    = 2000
SEED         = 42
TARGET_SCALE = 100.0   # học trên log-return × 100 (%) cho ổn định số học / regularization


def _search_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth":        trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 100, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
        "gamma":            trial.suggest_float("gamma", 1e-3, 5.0, log=True),
    }


def _model(params: dict, n_estimators: int, early_stop: bool = True) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=n_estimators, objective="reg:squarederror", tree_method="hist",
        eval_metric="mae", random_state=SEED, n_jobs=-1,
        early_stopping_rounds=EARLY_STOP if early_stop else None, **params,
    )


def _xy(df: pd.DataFrame, feats: list[str], target: str):
    return df[feats].to_numpy(dtype=float), df[target].to_numpy() * TARGET_SCALE


def _tune(train: pd.DataFrame, feats, target, folds, horizon) -> tuple[dict, float]:
    tune_folds = [purge_fold(train, f, horizon) for f in folds[-TUNE_FOLDS:]]

    def objective(trial):
        p, maes = _search_space(trial), []
        for f_tr, f_vl in tune_folds:
            X_tr, y_tr = _xy(f_tr, feats, target)
            X_vl, y_vl = _xy(f_vl, feats, target)
            m = _model(p, MAX_TREES).fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            maes.append(mean_absolute_error(y_vl, m.predict(X_vl)))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params, study.best_value


def _pred_frame(df, lr_pred, branch, horizon, split) -> pd.DataFrame:
    return prediction_frame(df, lr_pred, MODEL_NAME, branch, horizon, split)


def train_one(branch: str, horizon: int, folds: list[dict]) -> dict:
    t0 = time.time()
    splits, feats, target = make_dataset(branch, horizon)
    tr, va, te = splits["train"], splits["val"], splits["test"]

    best, cv_mae = _tune(tr, feats, target, folds, horizon)

    # Model Val: Train → early stopping theo Val
    X_tr, y_tr = _xy(tr, feats, target)
    X_va, y_va = _xy(va, feats, target)
    X_te, _    = _xy(te, feats, target)
    m_val = _model(best, MAX_TREES).fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    n_trees = int(m_val.best_iteration) + 1
    lr_val = m_val.predict(X_va) / TARGET_SCALE

    # Dự đoán out-of-fold trên mọi fold walk-forward (để hiệu chỉnh khoảng dự đoán)
    oof = []
    for fold in folds:
        f_tr, f_vl = purge_fold(tr, fold, horizon)
        if len(f_tr) < 250 or len(f_vl) < 50:
            continue
        m = _model(best, n_trees, early_stop=False).fit(*_xy(f_tr, feats, target), verbose=False)
        oof.append(_pred_frame(f_vl, m.predict(_xy(f_vl, feats, target)[0]) / TARGET_SCALE,
                               branch, horizon, "cv"))

    # Model Test: Train + Val, số cây cố định
    trva = pd.concat([tr, va])
    X_all, y_all = _xy(trva, feats, target)
    m_final = _model(best, n_trees, early_stop=False).fit(X_all, y_all, verbose=False)
    lr_test = m_final.predict(X_te) / TARGET_SCALE

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    m_final.save_model(MODEL_DIR / f"{branch}_h{horizon}.json")
    widths = fit_widths(pd.concat(oof), conditional=(branch == "kmeans"))
    with open(MODEL_DIR / f"{branch}_h{horizon}_intervals.json", "w", encoding="utf-8") as fh:
        json.dump({str(lv): {(r or "_all"): q for r, q in w.items()} for lv, w in widths.items()}, fh, indent=2)
    with open(MODEL_DIR / f"{branch}_h{horizon}_features.json", "w", encoding="utf-8") as fh:
        json.dump(feats, fh, ensure_ascii=False)

    preds = pd.concat([*oof,
                       _pred_frame(va, lr_val, branch, horizon, "val"),
                       _pred_frame(te, lr_test, branch, horizon, "test")])
    imp = pd.DataFrame({"model": MODEL_NAME, "branch": branch, "horizon": horizon,
                        "feature": feats, "importance_gain": m_final.feature_importances_})
    test_m = forecast_metrics(te["Gold Price"], te[f"target_price_h{horizon}"],
                              to_price(te, lr_test), te[target], lr_test)
    logger.info("%-6s h=%-2d | %d feats | trees=%4d | CV MAE(%%)=%.4f | Test MAE=%.2f MASE=%.4f DA=%.1f%% | %.0fs",
                branch, horizon, len(feats), n_trees, cv_mae, test_m["MAE"], test_m["MASE"],
                test_m["DA"], time.time() - t0)
    return dict(preds=preds, importance=imp,
                meta=dict(params=best, n_trees=n_trees, cv_mae_pct=cv_mae, n_features=len(feats),
                          rows={k: len(v) for k, v in splits.items()}))


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_DIR / "walk_forward_splits.json", encoding="utf-8") as fh:
        folds = json.load(fh)

    preds, imps, meta = [], [], {}
    for h in HORIZONS:
        for branch in BRANCHES:
            r = train_one(branch, h, folds)
            preds.append(r["preds"]); imps.append(r["importance"])
            meta[f"{branch}_h{h}"] = r["meta"]

    preds = pd.concat(preds, ignore_index=True)
    preds.to_csv(OUT_DIR / "predictions.csv", index=False)
    pd.concat(imps, ignore_index=True).to_csv(OUT_DIR / "feature_importance.csv", index=False)
    metrics = metrics_table(preds, MODEL_NAME)
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)

    metadata = {
        "model": MODEL_NAME, "xgboost_version": xgb.__version__,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tasks": BRANCHES, "clustering_algorithm": "kmeans", "horizons": HORIZONS,
        "dataset_version": "processed_v3", "target": "target_logret_h{h} (quy đổi ra giá khi đánh giá)",
        "tuning": f"Optuna TPE, {N_TRIALS} trials, {TUNE_FOLDS} walk-forward folds cuối (purged)",
        "final_fit": "Val: fit Train + early stopping; Test: refit Train+Val với số cây tốt nhất",
        "runs": meta,
    }
    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    logger.info("Test metrics:\n%s", metrics[metrics["split"] == "test"].round(4).to_string(index=False))
    logger.info("Results → %s", OUT_DIR)


# ── Dùng cho web app ───────────────────────────────────────────────────────
def predict_latest(history: pd.DataFrame, branch: str, horizon: int) -> dict:
    """
    history: dữ liệu dạng base_dataset (≥ 200 phiên gần nhất, có cột Date, Gold Price, *_logret, ...).
    Nhánh kmeans: gọi model K-Means để gán giai đoạn TRƯỚC, rồi mới dự đoán.
    """
    df = history.sort_values("Date").reset_index(drop=True)
    regime = None
    if branch == "kmeans":
        df = assign_regime(df)
        regime = df["market_phase"].iloc[-1]
    df, _ = build_features(df)
    feats = json.loads((MODEL_DIR / f"{branch}_h{horizon}_features.json").read_text(encoding="utf-8"))
    model = xgb.XGBRegressor(); model.load_model(MODEL_DIR / f"{branch}_h{horizon}.json")
    last = df.iloc[[-1]]
    lr = float(model.predict(last[feats].to_numpy(dtype=float))[0]) / TARGET_SCALE
    price_t = float(last["Gold Price"].iloc[0])
    widths = json.loads((MODEL_DIR / f"{branch}_h{horizon}_intervals.json").read_text(encoding="utf-8"))
    intervals = {}
    for lv, w in widths.items():
        q = w.get(regime, w["_all"]) if regime else w["_all"]
        intervals[lv] = (price_t * float(np.exp(lr - q)), price_t * float(np.exp(lr + q)))
    return {"date": last["Date"].iloc[0], "price_t": price_t,
            "pred_price": price_t * float(np.exp(lr)), "pred_logret": lr, "intervals": intervals,
            "regime": regime, "branch": branch, "horizon": horizon,
            "uses_regime_features": all(c in feats for c in REGIME_FEATURES)}


if __name__ == "__main__":
    run()
