from pathlib import Path

# Project root: config.py → preprocessing_v3 → source → DTMining
ROOT_DIR = Path(__file__).resolve().parents[2]

# ── Input ──────────────────────────────────────────────────────────────────
RAW_DATA = ROOT_DIR / "dataset" / "data_root" / "dataset_root.csv"

# ── Stage 1 output ─────────────────────────────────────────────────────────
OUT_DIR      = ROOT_DIR / "dataset" / "processed_v3"
BASE_DATASET = OUT_DIR / "base_dataset.csv"
REPORTS_DIR  = OUT_DIR / "reports"
FIGURES_DIR  = ROOT_DIR / "figures" / "v3"

# ── Stage 2 output ─────────────────────────────────────────────────────────
SPLIT_DIR  = OUT_DIR / "split"
MODELS_DIR = OUT_DIR / "models"

# ── Column definitions ──────────────────────────────────────────────────────
EVENTS = [
    "Economic Data", "logistic", "Politics", "oil", "OPEC",
    "Stock (US)", "Stock (Global)",
    "currency (Dollar)", "currency (Russian ruble)",
    "currency (Sterling pound)", "currency (Euro)", "currency (Yen)",
    "Inflation", "Job report", "unemployment rate",
    "CPI", "Interest Rate", "Bonds",
]

MARKET = [
    "Gold Price", "Silver", "Copper", "Oil", "Palladium", "Platinum",
    "Iron Price", "U.S dollar index", "DJI", "S&P 500",
]

MACRO = ["U.S Treasury", "Inflation Rate", "CPI Rate", "Fedral Funds Rate"]

# Log-return transform: ln(P_t / P_{t-1}), dùng cho cột giá luôn dương
LOGRET_COLS = [
    "Gold Price", "Silver", "Copper", "Palladium", "Platinum",
    "Iron Price", "U.S dollar index", "DJI", "S&P 500",
    "U.S Treasury", "CPI Rate",
]

# First-difference transform: P_t - P_{t-1}, dùng cho cột có thể âm hoặc bằng 0
DIFF_COLS = ["Oil", "Inflation Rate", "Fedral Funds Rate"]

# Giá gốc không dừng — giữ lại trong dataset (SARIMAX cần), nhưng không dùng làm feature ML
RAW_PRICE_COLS = set(MARKET + MACRO)

# ── Temporal split ─────────────────────────────────────────────────────────
TRAIN_END  = "2017-12-31"
VAL_START  = "2018-01-01"
VAL_END    = "2019-12-31"
TEST_START = "2020-01-01"

# ── Multi-horizon targets ──────────────────────────────────────────────────
HORIZONS = [1, 7, 30]   # đơn vị: ngày giao dịch (observations, không phải lịch)

# ── Walk-forward CV ────────────────────────────────────────────────────────
WF_MIN_TRAIN_YEARS = 3
WF_STEP_YEARS      = 1
WF_VAL_YEARS       = 1
