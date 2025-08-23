import scipy.sparse as sp

def to_sparse_matrix(data_file):
    users = data_file.ix[:,0].unique()
    users = users[users.argsort()]
    movies = data_file.ix[:,1].unique()
    movies = movies[movies.argsort()]

    number_of_rows = len(users)
    number_of_columns = len(movies)

    movie_indices, user_indices = {}, {}

    for i in range(len(movies)):
        movie_indices[movies[i]] = i

    for i in range(len(users)):
        user_indices[users[i]] = i

    #scipy sparse matrix to store the 1M matrix
    V = sp.lil_matrix((number_of_rows, number_of_columns))

    #adds data into the sparse matrix
    for index,row in data_file.iterrows():
        u, i , r  = row[0], row[1], row[2]
        V[user_indices[u], movie_indices[i]] = r

    return V #V.tocsr()

