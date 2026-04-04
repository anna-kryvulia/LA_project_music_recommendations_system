from data_loader import load_interactions
from preprocessing import clean_interactions, log_normalize_playcounts
from matrix_builder import create_id_mappings, build_interaction_matrix
from svd_model import SVDRecommender
from recommender import recommend_top_n


def main():
    df = load_interactions("data/user_artists.csv")

    df = clean_interactions(df)
    df = log_normalize_playcounts(df)

    user_to_index, artist_to_index, index_to_user, index_to_artist = create_id_mappings(df)

    interaction_matrix = build_interaction_matrix(df, user_to_index, artist_to_index)

    model = SVDRecommender(n_factors=20)
    model.fit(interaction_matrix)

    predicted_matrix = model.reconstruct_matrix()

    example_user_id = df["userID"].iloc[0]
    user_index = user_to_index[example_user_id]

    recommendations = recommend_top_n(
        original_matrix=interaction_matrix,
        predicted_matrix=predicted_matrix,
        user_index=user_index,
        index_to_artist=index_to_artist,
        n=10
    )

    print(f"Recommendations for user {example_user_id}:")
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    main()
