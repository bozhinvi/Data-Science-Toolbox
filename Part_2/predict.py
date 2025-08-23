import numpy as np 
# prediction function
def predict(user1, users, top_similarities):
    scores = np.zeros((users.shape[1],))

    for i in range(users.shape[1]):
        if i not in user1.nonzero()[1] and i in np.nonzero(users.getnnz(0))[0]:

            users_seen_the_movie = np.nonzero(users[:,i].getnnz(1))[0]
            for j in users_seen_the_movie:
                scores[i] += top_similarities[j] * (users[j,i] - users[j,:].mean())

            scores[i] =  user1.mean() + scores[i]/(sum(abs(top_similarities[users_seen_the_movie])))

    return scores


