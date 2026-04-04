import numpy as np
import pandas as pd

def clean_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates(subset=["userID", "artistID"])
    df = df[df["weight"] > 0].copy()
    return df

def log_normalize_playcounts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = np.log1p(df["weight"])
    return df
