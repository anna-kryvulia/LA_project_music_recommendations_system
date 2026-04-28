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

def sample_dataframe(df, n_users=500, n_artists=700, min_user_interactions=10, random_state=42):
    """
    Samples users and artists for faster testing.
    This function keeps users with enough interactions first.
    Then it keeps the most common artists inside the sampled user subset.
    """
    user_counts = df["userID"].value_counts()
    eligible_users = user_counts[user_counts >= min_user_interactions].index
    df_eligible = df[df["userID"].isin(eligible_users)].copy()
    sampled_users = (df_eligible["userID"].drop_duplicates().sample(n=min(n_users, len(eligible_users)), random_state=random_state))
    df_small = df_eligible[df_eligible["userID"].isin(sampled_users)].copy()
    top_artists = (df_small["artistID"].value_counts().head(n_artists).index)
    df_small = df_small[df_small["artistID"].isin(top_artists)].copy()

    # After filtering artists, some users may again have too few interactions.
    user_counts_after = df_small["userID"].value_counts()

    valid_users = user_counts_after[user_counts_after >= min_user_interactions].index

    df_small = df_small[df_small["userID"].isin(valid_users)].copy()

    return df_small
