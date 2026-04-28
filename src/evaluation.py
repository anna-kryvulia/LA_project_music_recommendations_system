import numpy as np
from recommender import recommend_top_n

def evaluate_recommender_at_k(train_matrix, predicted_matrix, test_df, user_to_index, artist_to_index, index_to_artist, k=10):
    """
    Precision: How many of the top-K recommendations are correct.
    Recall: How many hidden test artists were found in top-K.
    HitRate: Did the model find at least one hidden artist?
    """

    precisions = []
    recalls = []
    hit_rates = []

    for user_id in test_df["userID"].unique():
        if user_id not in user_to_index:
            continue

        user_test = test_df[test_df["userID"] == user_id]

        relevant_artists = set(user_test["artistID"])

        relevant_artists = {artist_id for artist_id in relevant_artists if artist_id in artist_to_index}

        if len(relevant_artists) == 0:
            continue

        user_index = user_to_index[user_id]

        recommendations = recommend_top_n(original_matrix=train_matrix, predicted_matrix=predicted_matrix, user_index=user_index, index_to_artist=index_to_artist, n=k)

        recommended_artists = {rec["artist_id"] for rec in recommendations}

        hits = recommended_artists.intersection(relevant_artists)

        precision = len(hits) / k
        recall = len(hits) / len(relevant_artists)
        hit_rate = 1 if len(hits) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        hit_rates.append(hit_rate)

    if len(precisions) == 0:
        return {
            "precision": 0,
            "recall": 0,
            "hit_rate": 0,
            "n_users_evaluated": 0}

    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "hit_rate": sum(hit_rates) / len(hit_rates),
        "n_users_evaluated": len(precisions)}


def evaluate_popularity_baseline_at_k(train_df, test_df, user_to_index, artist_to_index, k=10):
    """
    recommends globally most popular artists from train data.
    This is important because if SVD is not better than this,
    then SVD is not adding much personalization.
    """

    artist_popularity = (train_df.groupby("artistID")["weight"].sum().sort_values(ascending=False))

    popular_artists = list(artist_popularity.index)

    precisions = []
    recalls = []
    hit_rates = []

    for user_id in test_df["userID"].unique():
        if user_id not in user_to_index:
            continue

        user_test = test_df[test_df["userID"] == user_id]

        relevant_artists = set(user_test["artistID"])

        relevant_artists = {artist_id for artist_id in relevant_artists if artist_id in artist_to_index}

        if len(relevant_artists) == 0:
            continue

        user_train_artists = set(train_df[train_df["userID"] == user_id]["artistID"])

        recommendations = []

        for artist_id in popular_artists:
            if artist_id not in user_train_artists:
                recommendations.append(artist_id)

            if len(recommendations) == k:
                break

        recommended_artists = set(recommendations)

        hits = recommended_artists.intersection(relevant_artists)

        precision = len(hits) / k
        recall = len(hits) / len(relevant_artists)
        hit_rate = 1 if len(hits) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        hit_rates.append(hit_rate)

    if len(precisions) == 0:
        return {
            "precision": 0,
            "recall": 0,
            "hit_rate": 0,
            "n_users_evaluated": 0
        }

    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "hit_rate": sum(hit_rates) / len(hit_rates),
        "n_users_evaluated": len(precisions)
    }


def evaluate_random_baseline_at_k(train_df, test_df, user_to_index, artist_to_index, k=10, random_state=100):
    """
    recommends random artists that the user has not listened to in train data.
    SVD must be better than this. If it is not, the model is useless.
    """

    rng = np.random.default_rng(random_state)

    all_artists = set(artist_to_index.keys())

    precisions = []
    recalls = []
    hit_rates = []

    for user_id in test_df["userID"].unique():
        if user_id not in user_to_index:
            continue

        user_test = test_df[test_df["userID"] == user_id]

        relevant_artists = set(user_test["artistID"])

        relevant_artists = {artist_id for artist_id in relevant_artists if artist_id in artist_to_index}

        if len(relevant_artists) == 0:
            continue

        user_train_artists = set(train_df[train_df["userID"] == user_id]["artistID"])

        candidate_artists = list(all_artists - user_train_artists)

        if len(candidate_artists) == 0:
            continue

        n_recommendations = min(k, len(candidate_artists))

        recommendations = rng.choice(
            candidate_artists,
            size=n_recommendations,
            replace=False
        )

        recommended_artists = set(recommendations)

        hits = recommended_artists.intersection(relevant_artists)

        precision = len(hits) / k
        recall = len(hits) / len(relevant_artists)
        hit_rate = 1 if len(hits) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        hit_rates.append(hit_rate)

    if len(precisions) == 0:
        return {
            "precision": 0,
            "recall": 0,
            "hit_rate": 0,
            "n_users_evaluated": 0
        }

    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "hit_rate": sum(hit_rates) / len(hit_rates),
        "n_users_evaluated": len(precisions)
    }


def print_metrics(title, metrics, k=10):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Precision@{k}: {metrics['precision']:.4f}")
    print(f"Recall@{k}:    {metrics['recall']:.4f}")
    print(f"HitRate@{k}:   {metrics['hit_rate']:.4f}")
    print(f"Users tested:  {metrics['n_users_evaluated']}")
