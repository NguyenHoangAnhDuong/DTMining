
from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parent

MODELS = [
    "sarimax",
    "xgboost",
    "lightgbm",
    "prophet",
    "lstm",
]

PAGES = {
    "01_dashboard.py": "Dashboard",
    "02_dataset.py": "Dataset",
    "03_eda.py": "Exploratory Data Analysis",
    "04_preprocessing.py": "Data Preprocessing",
    "05_market_regime.py": "Market Regime Analysis",
    "06_forecast_results.py": "Forecast Results",
    "07_model_comparison.py": "Model Comparison",
    "08_reports.py": "Reports",
}

DIRECTORIES = [
    ".streamlit",
    ".vscode",
    "pages",
    "components",
    "src",
    "data/raw",
    "data/processed",
    "results/eda",
    "results/preprocessing",
    "results/clustering/kmeans",
    "results/clustering/gmm",
    "notebooks/common",
    "assets/images",
    "assets/styles",
    "docs",
]

for model in MODELS:
    DIRECTORIES.extend([
        f"results/forecasting/{model}/diagnostics",
        f"notebooks/{model}",
    ])


def create_file(relative_path, content=""):
    path = ROOT / relative_path

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Never overwrite existing files
    if path.exists():
        print(f"[SKIP] {relative_path}")
        return

    path.write_text(
        textwrap.dedent(content).lstrip("\n"),
        encoding="utf-8",
    )

    print(f"[FILE] {relative_path}")


# =====================================
# 1. CREATE DIRECTORIES
# =====================================

for directory in DIRECTORIES:
    folder = ROOT / directory

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[DIR] {directory}")


# =====================================
# 2. REQUIREMENTS
# =====================================

create_file(
    "requirements.txt",
    """
    streamlit>=1.36,<2
    pandas
    plotly
    """,
)


# =====================================
# 3. GITIGNORE
# =====================================

create_file(
    ".gitignore",
    """
    # Virtual environment
    .venv/
    venv/

    # Python cache
    __pycache__/
    *.py[cod]
    .pytest_cache/

    # Jupyter cache
    .ipynb_checkpoints/

    # Secrets
    .env
    .streamlit/secrets.toml

    # Operating system
    .DS_Store
    Thumbs.db

    # Raw dataset
    data/raw/*
    !data/raw/.gitkeep
    """,
)


# =====================================
# 4. STREAMLIT CONFIG
# =====================================

create_file(
    ".streamlit/config.toml",
    """
    [theme]
    primaryColor = "#D4AF37"
    backgroundColor = "#0F172A"
    secondaryBackgroundColor = "#1E293B"
    textColor = "#F8FAFC"
    font = "sans serif"

    [server]
    headless = true
    port = 8501

    [browser]
    gatherUsageStats = false
    """,
)


# =====================================
# 5. VSCODE CONFIG
# =====================================

settings = {
    "python.defaultInterpreterPath":
        "${workspaceFolder}/.venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": True,
    "editor.tabSize": 4,
    "files.encoding": "utf8",
    "files.exclude": {
        "**/__pycache__": True
    },
}

create_file(
    ".vscode/settings.json",
    json.dumps(settings, indent=4),
)


# =====================================
# 6. SOURCE FILES
# =====================================

create_file("src/__init__.py")

create_file(
    "src/constants.py",
    """
    from pathlib import Path

    ROOT_DIR = Path(__file__).resolve().parents[1]

    DATA_DIR = ROOT_DIR / "data"
    RESULTS_DIR = ROOT_DIR / "results"
    ASSETS_DIR = ROOT_DIR / "assets"

    APP_NAME = "Gold Intelligence"

    MODELS = [
        "sarimax",
        "xgboost",
        "lightgbm",
        "prophet",
        "lstm",
    ]

    MODEL_NAMES = {
        "sarimax": "SARIMAX",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "prophet": "Prophet",
        "lstm": "LSTM",
    }

    TASKS = {
        "task1": "With Clustering",
        "task2": "Without Clustering",
    }

    HORIZONS = [1, 7, 30]

    CLUSTERING_ALGORITHMS = [
        "kmeans",
        "gmm",
    ]

    METRICS = [
        "MAE",
        "RMSE",
        "MAPE",
        "R2",
        "MASE",
        "Directional_Accuracy",
    ]
    """,
)

create_file(
    "src/data_loader.py",
    """
    import pandas as pd

    from src.constants import DATA_DIR


    def load_raw_data():
        path = DATA_DIR / "raw" / "EGPBD.csv"

        if not path.exists():
            return pd.DataFrame()

        return pd.read_csv(path)


    def load_processed_data():
        path = (
            DATA_DIR
            / "processed"
            / "cleaned_data.csv"
        )

        if not path.exists():
            return pd.DataFrame()

        return pd.read_csv(path)
    """,
)

create_file(
    "src/result_loader.py",
    """
    import json
    import pandas as pd

    from src.constants import RESULTS_DIR, MODELS


    def load_model_results(model, filename):
        if model not in MODELS:
            raise ValueError("Invalid model name")

        if filename not in {
            "metrics.csv",
            "predictions.csv",
            "feature_importance.csv",
            "training_history.csv",
        }:
            raise ValueError("Invalid result filename")

        path = (
            RESULTS_DIR
            / "forecasting"
            / model
            / filename
        )

        if not path.exists():
            return pd.DataFrame()

        return pd.read_csv(path)


    def load_all_metrics():
        frames = []

        for model in MODELS:
            df = load_model_results(
                model,
                "metrics.csv",
            )

            if not df.empty:
                df["model"] = model
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(
            frames,
            ignore_index=True,
        )


    def load_metadata(model):
        if model not in MODELS:
            raise ValueError("Invalid model name")

        path = (
            RESULTS_DIR
            / "forecasting"
            / model
            / "metadata.json"
        )

        if not path.exists():
            return {}

        with path.open(encoding="utf-8") as file:
            return json.load(file)
    """,
)

create_file("src/utils.py")


# =====================================
# 7. COMPONENTS
# =====================================

for filename in [
    "__init__.py",
    "sidebar.py",
    "metric_cards.py",
    "filters.py",
    "charts.py",
]:
    create_file(f"components/{filename}")


# =====================================
# 8. STREAMLIT APPLICATION
# =====================================

create_file(
    "app.py",
    """
    import streamlit as st

    st.set_page_config(
        page_title="Gold Intelligence",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("GOLD INTELLIGENCE")

    st.sidebar.caption(
        "Market Regime & Gold Price Forecasting"
    )

    navigation = st.navigation({
        "OVERVIEW": [
            st.Page(
                "pages/01_dashboard.py",
                title="Dashboard",
                default=True,
            ),
        ],
        "DATA ANALYSIS": [
            st.Page(
                "pages/02_dataset.py",
                title="Dataset",
            ),
            st.Page(
                "pages/03_eda.py",
                title="EDA",
            ),
            st.Page(
                "pages/04_preprocessing.py",
                title="Preprocessing",
            ),
            st.Page(
                "pages/05_market_regime.py",
                title="Market Regime",
            ),
        ],
        "FORECASTING": [
            st.Page(
                "pages/06_forecast_results.py",
                title="Forecast Results",
            ),
            st.Page(
                "pages/07_model_comparison.py",
                title="Model Comparison",
            ),
            st.Page(
                "pages/08_reports.py",
                title="Reports",
            ),
        ],
    })

    st.sidebar.divider()
    st.sidebar.caption("Results Dashboard")
    st.sidebar.caption("Version 1.0.0")

    navigation.run()
    """,
)


# =====================================
# 9. DASHBOARD
# =====================================

create_file(
    "pages/01_dashboard.py",
    """
    import streamlit as st

    from src.constants import (
        MODELS,
        MODEL_NAMES,
        TASKS,
        HORIZONS,
        CLUSTERING_ALGORITHMS,
    )

    from src.result_loader import load_all_metrics

    st.title("Gold Intelligence Dashboard")

    st.caption(
        "Market Regime Analysis & "
        "Multi-Horizon Gold Price Forecasting"
    )

    cols = st.columns(4)

    cols[0].metric(
        "Forecast Models",
        len(MODELS),
    )

    cols[1].metric(
        "Experiment Tasks",
        len(TASKS),
    )

    cols[2].metric(
        "Forecast Horizons",
        len(HORIZONS),
    )

    cols[3].metric(
        "Clustering Algorithms",
        len(CLUSTERING_ALGORITHMS),
    )

    st.divider()

    st.subheader("Forecast Models")

    cols = st.columns(5)

    for col, model in zip(cols, MODELS):
        col.info(MODEL_NAMES[model])

    st.subheader("Experiment Tasks")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("### Task 1")
            st.write("With Clustering")
            st.caption(
                "Base Features + Cluster Features"
            )

    with right:
        with st.container(border=True):
            st.markdown("### Task 2")
            st.write("Without Clustering")
            st.caption("Base Features Only")

    st.subheader("Forecast Horizons")

    cols = st.columns(3)

    for col, horizon in zip(cols, HORIZONS):
        col.metric(
            "Horizon",
            f"h = {horizon}",
        )

    st.divider()

    st.subheader("Experiment Results")

    metrics = load_all_metrics()

    if metrics.empty:
        st.info(
            "No experiment results available. "
            "Please import results from Google Colab."
        )
    else:
        st.dataframe(
            metrics,
            use_container_width=True,
        )
    """,
)


# =====================================
# 10. OTHER PAGES
# =====================================

for filename, title in PAGES.items():

    if filename == "01_dashboard.py":
        continue

    content = f"""
    import streamlit as st

    st.title("{title}")

    st.info(
        "This page is under development."
    )
    """

    create_file(
        f"pages/{filename}",
        content,
    )


# =====================================
# 11. DOCUMENTATION
# =====================================

create_file(
    "README.md",
    """
    # Gold Intelligence

    Market Regime Analysis and
    Multi-Horizon Gold Price Forecasting.

    ## Architecture

    - Google Colab: Training and evaluation.
    - Streamlit: Results visualization.
    - CSV and JSON: Result storage.

    ## Forecast Models

    - SARIMAX
    - XGBoost
    - LightGBM
    - Prophet
    - LSTM

    ## Experiment Tasks

    - Task 1: With Clustering
    - Task 2: Without Clustering

    ## Forecast Horizons

    - h = 1
    - h = 7
    - h = 30

    ## Installation

        python -m pip install -r requirements.txt

    ## Run Application

        python -m streamlit run app.py

    ## Note

    This application displays precomputed results.
    It does not train models or generate new forecasts.
    """,
)

create_file(
    "docs/result_contract.md",
    """
    # Result Contract

    Each forecasting model exports:

    - metrics.csv
    - predictions.csv
    - metadata.json

    Optional files:

    - feature_importance.csv
    - training_history.csv
    - diagnostics/

    All results must identify:

    - Model
    - Task
    - Clustering algorithm
    - Forecast horizon
    - Dataset version
    """,
)

create_file("assets/styles/custom.css")


# =====================================
# 12. KEEP EMPTY DIRECTORIES
# =====================================

for directory in DIRECTORIES:
    folder = ROOT / directory

    if not any(folder.iterdir()):
        create_file(
            f"{directory}/.gitkeep"
        )


print("\n" + "=" * 50)
print("PROJECT INITIALIZED SUCCESSFULLY")
print("=" * 50)
print(f"Location: {ROOT}")
print("\nNext commands:")
print("python -m pip install -r requirements.txt")
print("python -m streamlit run app.py")
