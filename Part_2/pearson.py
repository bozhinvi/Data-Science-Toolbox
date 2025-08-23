import numpy as np

def pearson_correlation(user1, user2):

    norm_user2 = sum((np.array(user2.data[0]) - user2.mean())**2)
    norm_user1 = sum((np.array(user1.data[0]) - user1.mean())**2)
    normalization = np.sqrt(norm_user1 * norm_user2)
    common = set(user1.nonzero()[1]).intersection(user2.nonzero()[1])
    pearson_sum = 0

    for i in common:
        pearson_sum += (user1[:,i].data[0][0] - user1.mean()) * (user2[:,i].data[0][0] - user2.mean())

    return pearson_sum/normalization
