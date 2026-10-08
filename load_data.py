import os
import pandas as pd


def load_data(path: str, **kwargs) -> pd.DataFrame:
    """Läser in en csv-fil och returnerar den som en DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hittar inte {path}. Ligger csv-filerna i mappen Kaggle_Movie_Data?")

    # low_memory=False tar bort varningen om blandade datatyper i movies_metadata.csv
    df = pd.read_csv(path, low_memory=False, **kwargs)
    return df
