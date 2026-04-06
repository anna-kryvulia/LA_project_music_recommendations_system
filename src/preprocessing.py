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

def sample_dataframe(df, n_users=50, n_artists=50, min_user_interactions=5, random_state=42):
    user_counts = df["userID"].value_counts()
    eligible_users = user_counts[user_counts >= min_user_interactions].index
    sampled_users = (
        df[df["userID"].isin(eligible_users)]["userID"]
        .drop_duplicates()
        .sample(n=min(n_users, len(eligible_users)), random_state=random_state)
    )
    df_small = df[df["userID"].isin(sampled_users)].copy()
    top_artists = (
        df_small["artistID"]
        .value_counts()
        .head(n_artists)
        .index
    )
    df_small = df_small[df_small["artistID"].isin(top_artists)].copy()
    return df_small