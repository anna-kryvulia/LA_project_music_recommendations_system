import numpy as np


class SVDRecommender:
    """
    Manual SVD-based recommender.
    Important: this version converts the matrix to dense format.
    It is suitable for small/medium data or for educational purposes.
    """

    def __init__(self, n_factors: int = 20):
        self.n_factors = n_factors

        self.U = None
        self.Sigma = None
        self.Vt = None

        self.U_k = None
        self.Sigma_k = None
        self.Vt_k = None

        self.user_factors = None
        self.item_factors = None
        self.predicted_matrix = None

    def compute_svd(self, interaction_matrix):
        """
        Compute full SVD manually using eigendecomposition of A^T A.
        Returns:
            U     : left singular vectors
            Sigma : diagonal matrix of singular values
            Vt    : transpose of right singular vectors
        """
        A = interaction_matrix.toarray().astype(float)

        #A^T A
        ata = A.T @ A

        #eigh is used because A^T A is symmetric
        eigenvalues, eigenvectors = np.linalg.eigh(ata)

        # sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        #avoid tiny negative values
        eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
        singular_values = np.sqrt(eigenvalues)

        #keep only strictly positive singular values
        positive_mask = singular_values > 1e-10
        singular_values = singular_values[positive_mask]
        V = eigenvectors[:, positive_mask]

        #compute U from A V / sigma
        U_columns = []
        for i, sigma_i in enumerate(singular_values):
            v_i = V[:, i]
            u_i = (A @ v_i) / sigma_i

            norm_u_i = np.linalg.norm(u_i)
            if norm_u_i > 0:
                u_i = u_i / norm_u_i

            U_columns.append(u_i)

        U = np.column_stack(U_columns) if U_columns else np.empty((A.shape[0], 0))
        Sigma = np.diag(singular_values)
        Vt = V.T

        self.U = U
        self.Sigma = Sigma
        self.Vt = Vt

        return U, Sigma, Vt

    def truncate(self):
        """
        Keep top-k singular values and vectors.
        """
        if self.U is None or self.Sigma is None or self.Vt is None:
            raise ValueError("You must call compute_svd() before truncate().")

        max_rank = min(self.U.shape[1], self.Vt.shape[0], self.n_factors)
        if max_rank == 0:
            raise ValueError("SVD produced zero valid singular values.")

        self.U_k = self.U[:, :max_rank]
        self.Sigma_k = self.Sigma[:max_rank, :max_rank]
        self.Vt_k = self.Vt[:max_rank, :]

        return self.U_k, self.Sigma_k, self.Vt_k

    def build_latent_factors(self):
        """
        Build latent matrices:
            P = U_k @ sqrt(Sigma_k)
            Q = V_k @ sqrt(Sigma_k)
        """
        if self.U_k is None or self.Sigma_k is None or self.Vt_k is None:
            raise ValueError("You must call truncate() before build_latent_factors().")

        sigma_k_sqrt = np.sqrt(self.Sigma_k)

        self.user_factors = self.U_k @ sigma_k_sqrt
        self.item_factors = self.Vt_k.T @ sigma_k_sqrt

        return self.user_factors, self.item_factors

    def build_prediction_matrix(self):
        """
        Reconstruct dense recommendation matrix:
            R_hat = P @ Q^T
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("You must call build_latent_factors() before build_prediction_matrix().")

        self.predicted_matrix = self.user_factors @ self.item_factors.T
        return self.predicted_matrix

    def fit(self, interaction_matrix):
        """
        Full pipeline:
            1. compute_svd
            2. truncate
            3. build_latent_factors
            4. build_prediction_matrix
        """
        self.compute_svd(interaction_matrix)
        self.truncate()
        self.build_latent_factors()
        self.build_prediction_matrix()
        return self

    def reconstruct_matrix(self):
        """
        Return prediction matrix.
        """
        if self.predicted_matrix is None:
            raise ValueError("Model is not fitted yet.")
        return self.predicted_matrix
