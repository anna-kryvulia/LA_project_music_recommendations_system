from data_loader import load_interactions
from preprocessing import clean_interactions, log_normalize_playcounts, sample_dataframe
from matrix_builder import create_id_mappings, build_interaction_matrix
from svd_model import SVDRecommender
from recommender import recommend_top_n

def main():
    df = load_interactions("data/user_artists.csv")
    df = clean_interactions(df)
    df = log_normalize_playcounts(df)
    df = sample_dataframe(df, n_users=100,n_artists=300,min_user_interactions=10)

    user_to_index, artist_to_index, _, index_to_artist = create_id_mappings(df)
    interaction_matrix = build_interaction_matrix(df, user_to_index, artist_to_index)

    model = SVDRecommender(n_factors=30)
    model.fit(interaction_matrix)

    predicted_matrix = model.reconstruct_matrix()

    user_index = 0
    users = df["userID"].unique()

    for user in users[:10]:
        user_index = user_to_index[user]

        recommendations = recommend_top_n(
            original_matrix=interaction_matrix,
            predicted_matrix=predicted_matrix,
            user_index=user_index,
            index_to_artist=index_to_artist,
            n=10)

        print(f"\nRecommendations for user {user} (matrix index {user_index}):")
        for i, rec in enumerate(recommendations, start=1):
            print(
                f"{i}. artist_index={rec['artist_index']}, "
                f"artist_id={rec['artist_id']}, "
                f"score={rec['score']:.6f}")

if __name__ == "__main__":
    main()
