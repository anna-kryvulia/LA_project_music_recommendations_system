import pandas as pd

def train_test_split_by_user(df, test_size=0.2, min_interactions=10, random_state=1):
    """
    For each user, hide part of their listened artists into test set.
    train_df: data that the model will see
    test_df: hidden data that the model must predict
    Users with too few interactions are ignored because they are not useful for Top-N evaluation.
    """
    test_parts = []

    for _, user_df in df.groupby("userID"):
        if len(user_df) < min_interactions:
            continue

        n_test = max(1, int(len(user_df) * test_size))

        if len(user_df) - n_test < 3:
            continue

        user_test = user_df.sample(
            n=n_test,
            random_state=random_state)
        test_parts.append(user_test)

    if len(test_parts) == 0:
        raise ValueError(
            "No users available for train/test split. "
            "Try lowering min_interactions.")

    test_df = pd.concat(test_parts)
    train_df = df.drop(test_df.index)

    return train_df, test_df
