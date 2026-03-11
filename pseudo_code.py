# Pseudocode for algorithms mentioned

# Build user-item interaction matrix from listening data
function BuildInteractionMatrix(data):

    m = number of unique users in data
    n = number of unique artists in data

    initialize matrix R of size m x n with zeros

    for each record in data:
        user = record.user
        artist = record.artist
        count = record.listening_count
        R[user][artist] = count

    return R

# Compute SVD of matrix A 
function ComputeSVD(A):
    ATA = transpose(A) * A

    eigenvalues, eigenvectors = ComputeEigenDecomposition(ATA)

    SortEigenPairsDescending(eigenvalues, eigenvectors)

    singular_values = ComputeSingularValues(eigenvalues)

    V = eigenvectors
    U = ComputeLeftSingularVectors(A, V, singular_values)

    Sigma = BuildSigmaMatrix(A, singular_values)
    V_T = transpose(V)

    return U, Sigma, V_T


# Compute eigenvalues and eigenvectors of matrix M
function ComputeEigenDecomposition(M):
    solve det(M - lambda * I) = 0 for eigenvalues

    initialize empty list eigenvectors

    for each eigenvalue lambda_i:
        solve (M - lambda_i * I)v_i = 0
        normalize v_i
        add v_i to eigenvectors

    return eigenvalues, eigenvectors


# Sort eigenvalues and eigenvectors in descending order
function SortEigenPairsDescending(eigenvalues, eigenvectors):
    sort eigenvalues from largest to smallest
    reorder eigenvectors in the same order


# Compute singular values from eigenvalues
function ComputeSingularValues(eigenvalues):
    initialize empty list singular_values

    for each eigenvalue lambda_i:
        if lambda_i > 0:
            sigma_i = square root of lambda_i
        else:
            sigma_i = 0
        add sigma_i to singular_values

    return singular_values


# Compute left singular vectors
function ComputeLeftSingularVectors(A, V, singular_values):
    initialize matrix U with zeros

    for i from 1 to number of singular_values:
        if singular_values[i] > 0:
            u_i = (A * V[:, i]) / singular_values[i]
            normalize u_i
            put u_i into column i of U

    return U


# Build Sigma matrix of proper size
function BuildSigmaMatrix(A, singular_values):
    initialize matrix Sigma of size rows(A) x cols(A) with zeros

    for i from 1 to min(rows(A), cols(A)):
        Sigma[i][i] = singular_values[i]

    return Sigma



# Keep only the first k singular values and corresponding vectors
function SelectTopK(U, Sigma, V_T, k):
    U_k = first k columns of U
    Sigma_k = first k singular values of Sigma
    V_k_T = first k rows of V_T

    return U_k, Sigma_k, V_k_T


# Build latent factor matrices P and Q
function BuildMatricesPQ(U_k, Sigma_k, V_k_T):
    Sigma_k_sqrt = square root of Sigma_k

    P = U_k * Sigma_k_sqrt
    Q = transpose(V_k_T) * Sigma_k_sqrt

    return P, Q


# Predict score for one user-artist pair
function PredictScore(P, Q, user_id, artist_id):
    p_u = row user_id from P
    q_i = row artist_id from Q

    score = dot_product(p_u, q_i)

    return score

# Construct the recommendation matrix
function BuildRecommendationMatrix(P, Q):
    for user_id from 1 to number of users:
        for artist_id from 1 to number of artists:
            R_hat[user_id][artist_id] = PredictScore(P, Q, user_id, artist_id)


# Return top-N recommendations for a user
function RecommendItems(R, R_hat, user_id, N):
    initialize empty list recommendations

    for each item j:
        if R[user_id][j] == 0:
            score = R_hat[user_id][j]
            add (item j, score) to recommendations

    sort recommendations by score in descending order

    return first N items from recommendations