import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt
%matplotlib osx
import seaborn
import pylab

data = pd.read_csv('u.data', sep='\t', header = None, names = ['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1')
data.head(6)
data['movie_id'] = data['movie_id'] - 1

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movie_data = pd.read_csv('u.item', sep='|', names=m_cols, usecols=range(5), encoding='latin-1')

movie_data['movie_id'] = movie_data['movie_id'] -1

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('u.user', sep='|', names=u_cols,encoding='latin-1')

# create one merged DataFrame
movie_ratings = pd.merge(movie_data, data)
lens = pd.merge(movie_ratings, users)

most_rated = lens.groupby('title').size().sort_values(ascending=False)[:25]

most_rated.head()
most_rated.hist()
all_most_rated = lens.groupby('title').size().sort_values(ascending=False)

all_most_rated.hist()

all_most_rated.head()

rank = np.arange(all_most_rated.size)

sns.pointplot(rank, all_most_rated)


plt.plot(rank, all_most_rated)
plt.show()


movie_stats = lens.groupby('title').agg({'rating': [np.size, np.mean]})
movie_stats.head()


# sort by rating average
movie_stats.sort_values([('rating', 'mean')], ascending=False).head()



atleast_100 = movie_stats['rating']['size'] >= 100
movie_stats[atleast_100].sort_values([('rating', 'mean')], ascending=False)[:15]




# convert rating data into sparse user-item matrix
Y = to_sparse_matrix(data)
#compute similarity between two users
pearson_correlation(Y[1,:], Y[2,:])

# pick a user
user1 = Y[1,:]

# compute pearson_correlation over all users
similarities = np.zeros((Y.shape[0],))
for i in range(Y.shape[0]):
    similarities[i] = pearson_correlation(user1, Y[i,:])

# find top 30 most similar users
top_similar_indexes = similarities.argsort()[-31:-1]
users = Y[top_similar_indexes,:]
top_similarities = similarities[top_similar_indexes]
top_similarities.size

# get rating prediction based in these 30 similar users
predicted_ratings = predict(user1, users, top_similarities)

# top 10 highest movie rating predictions
top_10_index = predicted_ratings.argsort()[::-1][:10]

movie_data
movie_data.loc[movie_data['movie_id'].isin(top_10_index)]

# ratings for the top 10 highest predicted movies
predicted_ratings[top_10_index]


Y_movies = Y.transpose()

#pick a movie
movie179 = Y_movies[179,:]
movie_similarities = np.zeros((Y_movies.shape[0],))


for i in range(Y_movies.shape[0]):
    movie_similarities[i] = pearson_correlation(movie179, Y_movies[i,:])

movie_data.loc[movie_data['movie_id'].isin(top_sim_movies)]


R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]

R = np.array(R)

N = len(R)
M = len(R[0])
K = 2

P = np.random.rand(N,K)
Q = np.random.rand(M,K)

nP, nQ = matrix_factorization(R, P, Q, K)

np.dot(nP[0,:], nQ[0,:])



# prediction function
def predict(user1, users, top_similarities):
    scores = np.zeros((users.shape[1],))

    for i in range(users.shape[1]):
        if i not in user1.nonzero()[1] and i in np.nonzero(users.getnnz(0))[0]:

            users_seen_the_movie = np.nonzero(users[:,i].getnnz(1))[0]
            #compute scorring function
    return scores



def pearson_correlation(user1, user2):

    norm_user2 = sum((np.array(user2.data[0]) - user2.mean())**2)
    norm_user1 = sum((np.array(user1.data[0]) - user1.mean())**2)
    normalization = np.sqrt(norm_user1 * norm_user2)
    common = set(user1.nonzero()[1]).intersection(user2.nonzero()[1])
    pearson_sum = 0
    #compute pearson correlation!

    return pearson_sum/normalization


def to_sparse_matrix(data_file):
    users = data.ix[:,0].unique()
    users = users[users.argsort()]
    movies = data.ix[:,1].unique()
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




def matrix_factorization(R, P, Q, K, epochs=5000, eta=0.0002, beta=0.02):
    Q = Q.T
    for epoch in range(epochs):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:


        #eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    e = e + (beta/2) * ( sum(P[i,:]**2) + sum(Q[:,j]**2) )
        if e < 0.001:
            break
    return P, Q.T
