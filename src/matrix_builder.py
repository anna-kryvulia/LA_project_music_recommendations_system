from scipy.sparse import csr_matrix
#we are importing this library because the matrix is very sparse and normal numpy.array won't work good with it


def create_id_mappings(df):
    """
    Here we are tagging out users and artists with ID 
    (because ID which we got in database are not [0, n])
    So we are rebuilding their IDs to 
    real userID -> row index
    real artistID -> column index
    """
    user_ids = sorted(df["userID"].unique())
    artist_ids = sorted(df["artistID"].unique())

    #user_to_index: {2: 0, 5: 1...}
    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    artist_to_index = {artist_id: idx for idx, artist_id in enumerate(artist_ids)}

    #just mirroring: index_to_user: {0: 2, 1: 5...}
    index_to_user = {idx: user_id for user_id, idx in user_to_index.items()}
    index_to_artist = {idx: artist_id for artist_id, idx in artist_to_index.items()}

    return user_to_index, artist_to_index, index_to_user, index_to_artist


def build_interaction_matrix(df, user_to_index, artist_to_index):
    """
    This function builds the matrix 
    [13883, 11690, 0 ...],
    [500, 0, 0 ...],
    [0, 0, 0...] ...
    """
    row_indices = df["userID"].map(user_to_index).to_numpy()
    col_indices = df["artistID"].map(artist_to_index).to_numpy()
    values = df["weight"].to_numpy()

    matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(user_to_index), len(artist_to_index))
    )

    return matrix
