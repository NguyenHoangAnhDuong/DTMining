# XGBoost — Dự báo giá vàng ngày mai (Tomorrow gold price)
# Quy trình:
#   1. Load train/val/test đã scale từ bước tiền xử lý
#   2. Encode market_phase → int (-1=unknown, 0=Di ngang, 1=Tang, 2=Giam)
#   3. Tìm hyperparameter tối ưu bằng Optuna TPE + walk-forward CV (3 fold cuối)
#   4. Chạy walk-forward CV toàn bộ 14 folds với best params để có số liệu CV
#   5. Train final model: fit trên toàn Train, early-stop theo Val (2018-2019)
#   6. Đánh giá: MAE, RMSE, MAPE, R², MASE, Directional Accuracy
#   7. Lưu model (.pkl) và result_xgboost.md

import json
import logging
import os
import pickle
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = r"D:\University\Data Mining\Project_DTMining\DTMining"
DATA_DIR   = os.path.join(BASE_DIR, "dataset", "processed", "split")
MODEL_DIR  = os.path.join(BASE_DIR, "source", "model")
RESULT_DIR = os.path.join(BASE_DIR, "result")
WF_JSON    = os.path.join(DATA_DIR, "walk_forward_splits.json")

TRAIN_PATH = os.path.join(DATA_DIR, "train_scaled.csv")
VAL_PATH   = os.path.join(DATA_DIR, "val_scaled.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test_scaled.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET     = "Tomorrow gold price"
DATE_COL   = "Date"

# Cột không phải feature (loại khỏi X)
NON_FEATURES = frozenset({
    DATE_COL, TARGET,
    "Next week gold price", "Next Month gold price",
    "market_phase",
})

MARKET_PHASE_MAP = {"Di ngang": 0, "Tang": 1, "Giam": 2}

OPTUNA_N_TRIALS   = 50
OPTUNA_N_FOLDS    = 3    # Dùng 3 fold cuối cho tuning
EARLY_STOP_ROUNDS = 50
SEED              = 42


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def encode_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột market_phase_enc (int) và trả về bản copy."""
    df = df.copy()
    df["market_phase_enc"] = (
        df["market_phase"].map(MARKET_PHASE_MAP).fillna(-1).astype(int)
    )
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Trả về danh sách cột feature theo thứ tự cột trong df."""
    base = [c for c in df.columns if c not in NON_FEATURES]
    return base + ["market_phase_enc"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def naive_mae(y: np.ndarray) -> float:
    """MAE của naive random-walk: dự báo ngày mai = giá hôm nay."""
    return float(mean_absolute_error(y[1:], y[:-1]))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, nm: float) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # Tránh chia cho 0 khi giá = 0 (không xảy ra với giá vàng thực tế)
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    r2   = float(r2_score(y_true, y_pred))
    mase = float(mae / nm) if nm > 1e-8 else float("nan")

    # Directional Accuracy: tỷ lệ dự báo đúng chiều giá lên/xuống
    # Dùng giá thực ngày i làm baseline để xác định chiều dự báo ngày i+1
    prev  = y_true[:-1]
    act_d = np.sign(y_true[1:] - prev)
    prd_d = np.sign(y_pred[1:] - prev)
    da    = float(np.mean(act_d == prd_d) * 100)

    return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2, MASE=mase, DA=da)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_xgb(params: dict, n_estimators: int = 1000) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators          = n_estimators,
        learning_rate         = params["learning_rate"],
        max_depth             = params["max_depth"],
        min_child_weight      = params["min_child_weight"],
        subsample             = params["subsample"],
        colsample_bytree      = params["colsample_bytree"],
        reg_alpha             = params["reg_alpha"],
        reg_lambda            = params["reg_lambda"],
        objective             = "reg:squarederror",
        tree_method           = "hist",
        random_state          = SEED,
        n_jobs                = -1,
        eval_metric           = "mae",
        early_stopping_rounds = EARLY_STOP_ROUNDS,
    )


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------

def _cv_fold_mae(
    train_df: pd.DataFrame,
    fold: dict,
    feat_cols: list[str],
    params: dict,
) -> float:
    dates = pd.to_datetime(train_df[DATE_COL])
    f_tr  = train_df[dates <= pd.Timestamp(fold["train_end"])].copy()
    f_vl  = train_df[
        (dates >= pd.Timestamp(fold["val_start"])) &
        (dates <= pd.Timestamp(fold["val_end"]))
    ].copy()

    if len(f_tr) < 100 or len(f_vl) < 10:
        return float("inf")

    f_tr = encode_phase(f_tr)
    f_vl = encode_phase(f_vl)

    mdl = _build_xgb(params)
    mdl.fit(
        f_tr[feat_cols].values, f_tr[TARGET].values,
        eval_set=[(f_vl[feat_cols].values, f_vl[TARGET].values)],
        verbose=False,
    )
    y_pred = mdl.predict(f_vl[feat_cols].values)
    return float(mean_absolute_error(f_vl[TARGET].values, y_pred))


def walk_forward_cv(
    train_df: pd.DataFrame,
    folds: list[dict],
    feat_cols: list[str],
    params: dict,
) -> list[dict]:
    results = []
    for fold in folds:
        dates = pd.to_datetime(train_df[DATE_COL])
        f_tr  = train_df[dates <= pd.Timestamp(fold["train_end"])].copy()
        f_vl  = train_df[
            (dates >= pd.Timestamp(fold["val_start"])) &
            (dates <= pd.Timestamp(fold["val_end"]))
        ].copy()

        if len(f_tr) < 100 or len(f_vl) < 10:
            continue

        f_tr = encode_phase(f_tr)
        f_vl = encode_phase(f_vl)

        mdl = _build_xgb(params)
        mdl.fit(
            f_tr[feat_cols].values, f_tr[TARGET].values,
            eval_set=[(f_vl[feat_cols].values, f_vl[TARGET].values)],
            verbose=False,
        )
        y_pred = mdl.predict(f_vl[feat_cols].values)
        y_true = f_vl[TARGET].values
        nm     = naive_mae(y_true)
        m      = compute_metrics(y_true, y_pred, nm)
        m["fold"] = fold["fold"]
        results.append(m)
        logger.info(
            "Fold %2d  MAE=%8.2f  RMSE=%8.2f  MAPE=%5.2f%%  R²=%7.4f  MASE=%.4f  DA=%5.1f%%",
            fold["fold"], m["MAE"], m["RMSE"], m["MAPE"], m["R2"], m["MASE"], m["DA"],
        )
    return results


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _write_md(
    best_params: dict,
    best_iter: int,
    n_feat: int,
    cv_df: pd.DataFrame,
    cv_mean: pd.Series,
    val_m: dict,
    test_m: dict,
    top10: pd.Series,
) -> None:
    lines = [
        "# Kết quả mô hình XGBoost — Dự báo giá vàng ngày mai",
        "",
        f"> Ngày chạy: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"> XGBoost {xgb.__version__}",
        "",
        "---",
        "",
        "## 1. Cấu hình mô hình",
        "",
        "| Tham số | Giá trị |",
        "|---|---|",
        "| Target | `Tomorrow gold price` |",
        f"| Số feature | {n_feat} |",
        f"| Số estimators tốt nhất | {best_iter} |",
        f"| Learning rate | {best_params['learning_rate']:.5f} |",
        f"| Max depth | {best_params['max_depth']} |",
        f"| Min child weight | {best_params['min_child_weight']} |",
        f"| Subsample | {best_params['subsample']:.4f} |",
        f"| Colsample bytree | {best_params['colsample_bytree']:.4f} |",
        f"| Reg alpha (L1) | {best_params['reg_alpha']:.6f} |",
        f"| Reg lambda (L2) | {best_params['reg_lambda']:.6f} |",
        f"| Hyperparameter tuning | Optuna TPE — {OPTUNA_N_TRIALS} trials, {OPTUNA_N_FOLDS} WF folds cuối |",
        "",
        "---",
        "",
        "## 2. Kết quả Walk-Forward Cross-Validation (trên tập Train 2001–2017, 14 folds)",
        "",
        "| Fold | MAE | RMSE | MAPE (%) | R² | MASE | Directional Accuracy (%) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in cv_df.iterrows():
        lines.append(
            f"| {int(row['fold'])} "
            f"| {row['MAE']:.2f} "
            f"| {row['RMSE']:.2f} "
            f"| {row['MAPE']:.2f} "
            f"| {row['R2']:.4f} "
            f"| {row['MASE']:.4f} "
            f"| {row['DA']:.1f} |"
        )

    lines += [
        f"| **TB** "
        f"| **{cv_mean['MAE']:.2f}** "
        f"| **{cv_mean['RMSE']:.2f}** "
        f"| **{cv_mean['MAPE']:.2f}** "
        f"| **{cv_mean['R2']:.4f}** "
        f"| **{cv_mean['MASE']:.4f}** "
        f"| **{cv_mean['DA']:.1f}** |",
        "",
        "---",
        "",
        "## 3. Kết quả trên tập Validation (2018–2019)",
        "",
        "| Metric | Giá trị |",
        "|---|---|",
        f"| MAE | {val_m['MAE']:.4f} USD/oz |",
        f"| RMSE | {val_m['RMSE']:.4f} USD/oz |",
        f"| MAPE | {val_m['MAPE']:.4f}% |",
        f"| R² | {val_m['R2']:.4f} |",
        f"| MASE | {val_m['MASE']:.4f} |",
        f"| Directional Accuracy | {val_m['DA']:.2f}% |",
        "",
        "---",
        "",
        "## 4. Kết quả trên tập Test (2020–2022)",
        "",
        "| Metric | Giá trị |",
        "|---|---|",
        f"| MAE | {test_m['MAE']:.4f} USD/oz |",
        f"| RMSE | {test_m['RMSE']:.4f} USD/oz |",
        f"| MAPE | {test_m['MAPE']:.4f}% |",
        f"| R² | {test_m['R2']:.4f} |",
        f"| MASE | {test_m['MASE']:.4f} |",
        f"| Directional Accuracy | {test_m['DA']:.2f}% |",
        "",
        "---",
        "",
        "## 5. Top 10 Feature Importance (gain)",
        "",
        "| Rank | Feature | Importance |",
        "|---:|---|---:|",
    ]

    for rank, (feat, imp) in enumerate(top10.items(), 1):
        lines.append(f"| {rank} | `{feat}` | {imp:.6f} |")

    lines += [
        "",
        "---",
        "",
        "## 6. Chú giải metric",
        "",
        "| Metric | Mô tả |",
        "|---|---|",
        "| **MAE** | Sai số tuyệt đối trung bình (USD/oz) — đơn vị giá vàng |",
        "| **RMSE** | Căn bậc hai sai số bình phương trung bình (USD/oz) — phạt sai số lớn nhiều hơn |",
        "| **MAPE** | Sai số phần trăm tuyệt đối trung bình (%) |",
        "| **R²** | Hệ số xác định — càng gần 1 mô hình giải thích variance càng tốt |",
        "| **MASE** | MAE chuẩn hóa bởi naive random-walk; **MASE < 1** có nghĩa tốt hơn baseline |",
        "| **DA** | Directional Accuracy — % ngày dự báo đúng chiều tăng/giảm so với giá thực ngày trước |",
        "",
        "- **Naive baseline**: dự báo ngày mai = giá đóng cửa hôm nay (random walk)",
        "- **Tập test 2020–2022**: bao gồm giai đoạn COVID-19 và chu kỳ tăng mạnh của giá vàng",
        "- **Model**: `source/model/xgboost_model.pkl`",
    ]

    md_path = os.path.join(RESULT_DIR, "result_xgboost.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.info("Report saved → %s", md_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> tuple:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # --- Load data ---
    logger.info("Đọc dữ liệu …")
    train = pd.read_csv(TRAIN_PATH)
    val   = pd.read_csv(VAL_PATH)
    test  = pd.read_csv(TEST_PATH)
    logger.info("Train %d | Val %d | Test %d hàng", len(train), len(val), len(test))

    with open(WF_JSON, encoding="utf-8") as fh:
        wf_folds = json.load(fh)
    logger.info("Walk-forward folds: %d", len(wf_folds))

    # Chuẩn bị feature cols (tính trước để nhất quán qua các fold)
    train_enc = encode_phase(train)
    feat_cols = get_feature_cols(train_enc)
    logger.info("Số feature: %d", len(feat_cols))

    # --- Optuna: tìm best hyperparameters trên 3 fold cuối ---
    tune_folds = wf_folds[-OPTUNA_N_FOLDS:]
    logger.info(
        "Optuna tuning — %d trials, folds %d–%d …",
        OPTUNA_N_TRIALS, tune_folds[0]["fold"], tune_folds[-1]["fold"],
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate"   : trial.suggest_float("learning_rate",    0.005, 0.15, log=True),
            "max_depth"       : trial.suggest_int(  "max_depth",        3, 9),
            "min_child_weight": trial.suggest_int(  "min_child_weight", 1, 10),
            "subsample"       : trial.suggest_float("subsample",        0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha"       : trial.suggest_float("reg_alpha",        1e-4, 10.0, log=True),
            "reg_lambda"      : trial.suggest_float("reg_lambda",       0.05, 10.0, log=True),
        }
        fold_maes = [
            _cv_fold_mae(train, fold, feat_cols, params)
            for fold in tune_folds
        ]
        return float(np.mean(fold_maes))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    logger.info("Best hyperparams: %s", best_params)
    logger.info("Best tune-folds MAE: %.4f", study.best_value)

    # --- Walk-forward CV toàn bộ 14 folds với best params ---
    logger.info("Walk-forward CV (14 folds) …")
    cv_results = walk_forward_cv(train, wf_folds, feat_cols, best_params)
    cv_df      = pd.DataFrame(cv_results)
    cv_mean    = cv_df.drop(columns="fold").mean()
    logger.info(
        "CV avg → MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  R²=%.4f  MASE=%.4f  DA=%.1f%%",
        cv_mean["MAE"], cv_mean["RMSE"], cv_mean["MAPE"],
        cv_mean["R2"],  cv_mean["MASE"], cv_mean["DA"],
    )

    # --- Train final model: fit trên Train, early-stop trên Val ---
    logger.info("Training final model (Train + early stop on Val) …")
    val_enc  = encode_phase(val)
    test_enc = encode_phase(test)

    X_train, y_train = train_enc[feat_cols].values, train_enc[TARGET].values
    X_val,   y_val   = val_enc[feat_cols].values,   val_enc[TARGET].values
    X_test,  y_test  = test_enc[feat_cols].values,  test_enc[TARGET].values

    final_model = _build_xgb(best_params, n_estimators=2000)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=200,
    )
    best_iter = int(final_model.best_iteration) + 1
    logger.info("Best iteration: %d", best_iter)

    # --- Evaluate ---
    pred_val  = final_model.predict(X_val)
    pred_test = final_model.predict(X_test)

    val_m  = compute_metrics(y_val,  pred_val,  naive_mae(y_val))
    test_m = compute_metrics(y_test, pred_test, naive_mae(y_test))

    logger.info(
        "Validation  → MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  R²=%.4f  MASE=%.4f  DA=%.1f%%",
        val_m["MAE"], val_m["RMSE"], val_m["MAPE"], val_m["R2"], val_m["MASE"], val_m["DA"],
    )
    logger.info(
        "Test        → MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  R²=%.4f  MASE=%.4f  DA=%.1f%%",
        test_m["MAE"], test_m["RMSE"], test_m["MAPE"], test_m["R2"], test_m["MASE"], test_m["DA"],
    )

    # --- Save model ---
    model_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump(final_model, fh)
    logger.info("Model saved → %s", model_path)

    # --- Feature importance (top 10 by gain) ---
    importance = pd.Series(
        final_model.feature_importances_, index=feat_cols
    ).sort_values(ascending=False)
    top10 = importance.head(10)

    # --- Markdown report ---
    _write_md(best_params, best_iter, len(feat_cols), cv_df, cv_mean, val_m, test_m, top10)

    return final_model, val_m, test_m


if __name__ == "__main__":
    main()
