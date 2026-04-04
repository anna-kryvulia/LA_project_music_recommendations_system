import numpy as np
from sklearn.metrics import mean_squared_error


def compute_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def precision_at_k(recommended_items, relevant_items, k: int) -> float:
    recommended_top_k = recommended_items[:k]
    hits = sum(1 for item in recommended_top_k if item in relevant_items)
    return hits / k if k > 0 else 0.0