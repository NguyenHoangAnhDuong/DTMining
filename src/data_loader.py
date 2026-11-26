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
