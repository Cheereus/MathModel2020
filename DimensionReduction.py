import numpy as np
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, MaxAbsScaler


# Normalize
def get_normalize(data):
    prepress = MinMaxScaler()
    X = prepress.fit_transform(data)
    return X


# t-SNE
def t_SNE(data, dim=3, perp=30, with_normalize=False):
    if with_normalize:
        data = get_normalize(data)

    data = np.array(data)
    tsne = TSNE(n_components=dim, init='pca', perplexity=perp, method='exact')
    tsne.fit_transform(data)

    return tsne.embedding_


def Isometric(data, n_neighbors=5, n_components=2):
    return Isomap(n_neighbors=n_neighbors, n_components=n_components).fit_transform(data)


# PCA
def get_pca(data, c=3, with_normalize=False):
    if with_normalize:
        X = get_normalize(data)

    pca_result = PCA(n_components=c)
    pca_result.fit(data)
    newX = pca_result.fit_transform(data)

    return newX, pca_result.explained_variance_ratio_, pca_result
