import numpy as np

def recommend_top_n(
    original_matrix,
    predicted_matrix,
    user_index: int,
    index_to_artist: dict,
    n: int = 10
):
    user_interactions = original_matrix[user_index].toarray().ravel()
    user_scores = predicted_matrix[user_index].copy()

    user_scores[user_interactions > 0] = -np.inf

    top_indices = np.argsort(user_scores)[::-1][:n]

    recommendations = [
        {
            "artist_index": int(idx),
            "artist_id": int(index_to_artist[idx]),
            "score": float(user_scores[idx]),
        }
        for idx in top_indices
        if np.isfinite(user_scores[idx])
    ]

    return recommendations