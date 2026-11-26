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
