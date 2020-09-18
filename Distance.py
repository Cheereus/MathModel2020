from scipy.spatial.distance import pdist
import numpy as np


def cosine_matrix(data):

    n_sample, n_feature = data.shape
    dis = np.zeros((n_sample, n_sample))

    for i in range(n_sample):
        for j in range(n_sample):
            dis[i][j] = pdist(np.vstack([data[i], data[j]]), 'cosine')
            dis[j][i] = pdist(np.vstack([data[i], data[j]]), 'cosine')

    return dis
