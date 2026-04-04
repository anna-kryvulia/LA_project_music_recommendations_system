import numpy as np
from scipy.sparse.linalg import svds

class SVDRecommender:
    """
    We are creating the class because it is easier to save n-factors
    Also it is easier to call functions .fit, .reconstruct_matrix
    """
    def __init__(self, n_factors: int = 20):
        self.n_factors = n_factors
        self.predicted_matrix = None

    def fit(self, interaction_matrix):
        """
        This function's taking the matrix userXartist
        Doing svd (build-in) function + sorting the components
        Building of the latent representation
        Predicted matrix
        """
        #truncated svd
        U, sigma, Vt = svds(interaction_matrix, k=self.n_factors)
        #left singular vectors, singular values, right singular vectors
        idx = np.argsort(sigma)[::-1]

        U = U[:, idx]
        sigma = sigma[idx]
        Vt = Vt[idx, :]
        sigma_sqrt = np.diag(np.sqrt(sigma))

        #building factors
        user_factors = U @ sigma_sqrt
        item_factors = Vt.T @ sigma_sqrt

        self.predicted_matrix = user_factors @ item_factors.T
        return self

    def reconstruct_matrix(self):
        """
        Building the prediction matrix
        """
        return self.predicted_matrix
