from data_loader import load_interactions
from preprocessing import clean_interactions, sample_dataframe, log_normalize_playcounts
from matrix_builder import create_id_mappings, build_interaction_matrix
from svd_model import SVDRecommender
from recommender import recommend_top_n
from train_test_splitting import train_test_split_by_user
from evaluation import evaluate_recommender_at_k, evaluate_popularity_baseline_at_k, evaluate_random_baseline_at_k, print_metrics
from model_selection import choose_best_k


def print_user_examples(train_df, test_df, train_matrix, predicted_matrix, user_to_index, index_to_artist, n_users=5, k=10):
    """
    Prints examples of hidden artists and recommended artists.
    This helps to understand what the model actually predicts.
    """

    print("\nEXAMPLES")

    shown = 0

    for user_id in test_df["userID"].unique():
        if user_id not in user_to_index:
            continue

        user_index = user_to_index[user_id]

        recommendations = recommend_top_n(
            original_matrix=train_matrix,
            predicted_matrix=predicted_matrix,
            user_index=user_index,
            index_to_artist=index_to_artist,
            n=k)

        hidden_artists = set(test_df[test_df["userID"] == user_id]["artistID"])

        recommended_artists = {rec["artist_id"] for rec in recommendations}

        hits = recommended_artists.intersection(hidden_artists)

        train_artists_count = len(train_df[train_df["userID"] == user_id]["artistID"].unique())

        print(f"\nUser {user_id}")
        print(f"Train artists count: {train_artists_count}")
        print(f"Hidden test artists: {hidden_artists}")
        print(f"Recommended artists: {recommended_artists}")
        print(f"Correctly predicted: {hits}")

        shown += 1

        if shown >= n_users:
            break


def test_model():
    k = 10

    df = load_interactions("data/user_artists.csv")

    df = clean_interactions(df)
    df = log_normalize_playcounts(df)
    df = sample_dataframe(df, n_users=500,n_artists=700,min_user_interactions=10)

    train_df, test_df = train_test_split_by_user(
        df,
        test_size=0.2,
        min_interactions=10
    )

    print("\nTRAIN / TEST SPLIT")
    print("==================")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Train users: {train_df['userID'].nunique()}")
    print(f"Test users: {test_df['userID'].nunique()}")
    print(f"Train artists: {train_df['artistID'].nunique()}")
    print(f"Test artists: {test_df['artistID'].nunique()}")

    user_to_index, artist_to_index, _, index_to_artist = create_id_mappings(train_df)

    train_matrix = build_interaction_matrix(
        train_df,
        user_to_index,
        artist_to_index
    )

    best_k, _ = choose_best_k(
    recommender_class=SVDRecommender,
    train_matrix=train_matrix,
    test_df=test_df,
    user_to_index=user_to_index,
    artist_to_index=artist_to_index,
    index_to_artist=index_to_artist,
    evaluate_function=evaluate_recommender_at_k,
    k_values=[5, 10, 15, 20, 30, 40, 50],
    top_n=k)

    model = SVDRecommender(n_factors=best_k)
    model.fit(train_matrix)

    predicted_matrix = model.reconstruct_matrix()

    svd_metrics = evaluate_recommender_at_k(
        train_matrix=train_matrix,
        predicted_matrix=predicted_matrix,
        test_df=test_df,
        user_to_index=user_to_index,
        artist_to_index=artist_to_index,
        index_to_artist=index_to_artist,
        k=k)

    popularity_metrics = evaluate_popularity_baseline_at_k(
        train_df=train_df,
        test_df=test_df,
        user_to_index=user_to_index,
        artist_to_index=artist_to_index,
        k=k)

    random_metrics = evaluate_random_baseline_at_k(
        train_df=train_df,
        test_df=test_df,
        user_to_index=user_to_index,
        artist_to_index=artist_to_index,
        k=k)

    print("\nMODEL EVALUATION")
    print("================")

    print_metrics("SVD MODEL", svd_metrics, k=k)
    print_metrics("POPULARITY BASELINE", popularity_metrics, k=k)
    print_metrics("RANDOM BASELINE", random_metrics, k=k)

    print("\nINTERPRETATION")
    print("==============")

    if svd_metrics["hit_rate"] > random_metrics["hit_rate"]:
        print("SVD is better than random baseline.")
    else:
        print("SVD is NOT better than random baseline. The model is basically useless.")

    if svd_metrics["hit_rate"] > popularity_metrics["hit_rate"]:
        print("SVD is better than popularity baseline.")
    else:
        print("SVD is NOT better than popularity baseline. The model mostly fails to personalize.")

    if svd_metrics["precision"] < 0.05:
        print(
            "Precision@10 is low. This is expected for sparse implicit-feedback data, "
            "especially when each user has only a few hidden test artists."
        )

    print_user_examples(
        train_df=train_df,
        test_df=test_df,
        train_matrix=train_matrix,
        predicted_matrix=predicted_matrix,
        user_to_index=user_to_index,
        index_to_artist=index_to_artist,
        n_users=5,
        k=k)


if __name__ == "__main__":
    test_model()