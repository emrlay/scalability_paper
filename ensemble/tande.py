"""
Efficient version of nystroem sampling-based spectral clustering.
Code by Zhijie Lin.
First version: 3rd Dec, 2017.
Last Modified: 30th Dec, 2017.
----------------------------------------------------------------
Reference:
Mu Li, Xiao-Chen Lian, James T. Kwok, Bao-Liang Lu
Time and Space Efficient Spectral Clustering via Column Sampling
Proceedings of the 2011 IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2011)
"""
from __future__ import print_function
import numpy as np
from numpy import linalg as LA
from scipy import sparse
from sklearn import cluster
from sklearn import preprocessing


def _build_hyperedge_affnity_matrix(features, additional=None):
    if additional is None:
        if sparse.isspmatrix_csr(features):
            return sparse.csr_matrix.dot(features, features.transpose().tocsr()).todense().astype(np.float64)
        else:
            return np.dot(features, features.transpose()).astype(np.float64)
    else:
        if sparse.isspmatrix_csr(features):
            return sparse.csr_matrix.dot(sparse.csr_matrix.dot(features, additional).tocsr(),
                                         features.transpose().tocsr()).todense().astype(np.float64)
        else:
            return np.dot(np.dot(features, additional), features.transpose())


def _affinity_vector_hyperedge(feature_1, features, additional=None):
    if additional is None:
        if sparse.isspmatrix_csr(features):
            return sparse.csr_matrix.dot(feature_1, features.transpose().tocsr()).todense().astype(np.float64)
        else:
            return np.dot(feature_1, features.transpose()).astype(np.float64)
    else:
        return np.dot(np.dot(feature_1, additional), features.transpose())


def _affinity_vectors_hyperedge(feature_block, features, additional=None):
    if additional is None:
        if sparse.isspmatrix_csr(features):
            return sparse.csr_matrix.dot(feature_block, features.transpose().tocsr()).todense().astype(np.float64)
        else:
            return np.dot(feature_block, features.transpose()).astype(np.float64)
    else:
        return None

_affinity_methods = {"hyperedge": _build_hyperedge_affnity_matrix}
_affinity_vec_methods = {"hyperedge": _affinity_vector_hyperedge}


def build_affinity_matrix(features, method='', additional=None):
    return _affinity_methods[method](features, additional=additional)


def efficient_spectral_clustering(features, k, additional=None, m=1000, method='hyperedge', interval=1000,
                                  release=0, verbose=False):
    """
    time and space efficient spectral clustering algorithm by Li

    :param features: feature vectors
    :param k: number of clusters
    :param additional: additional weighting vector
    :param m: number of sampled points, default to 1000.
    :param method: method used to generated affinity matrix, default to 'hyperedge'
    :return:
    """
    n_instances = features.shape[0]

    # sample for m points (input data)
    selected_indices = np.random.choice(n_instances, m, replace=False)
    selected_features = features[selected_indices, :]

    # build a_one_one, d_star and m_star (algorithm line 1~3)
    a_one_one = build_affinity_matrix(selected_features, additional=additional, method=method)
    scaler = np.max(a_one_one)
    a_one_one /= scaler
    a_one_one = np.squeeze(np.asarray(a_one_one))
    if verbose:
        print ('[ESC] a_one_one is derived. size is '+str(a_one_one.shape))

    d_star = np.diag(np.sum(a_one_one, axis=1) ** -0.5)
    m_star = np.dot(np.dot(d_star, a_one_one), d_star)
    if verbose:
        print ('[ESC] Matrix M* is derived. size is '+str(m_star.shape))

    # eigenvalue decomposition (algorithm line 4, first part)
    eigenvalues, eigenvecs = LA.eig(m_star)
    if verbose:
        print ('[ESC] Eigenvalue Decomposition Completed.')
        print (eigenvalues)
        print (eigenvalues.real)

    # keep k-largest eigenvalues/eigen vectors (algorithm line 4, second part)
    # diag_eigen_values --> matrix GAMMA ** (-1)
    # keep_eigenvecs --> V
    first_k_indices = np.argsort(-eigenvalues)
    if verbose:
        print (eigenvalues[first_k_indices[1:k+1+release]])
    diag_eigen_values = np.diag(eigenvalues.real[first_k_indices[1:k+1+release]] ** (-1))
    keep_eigenvecs = eigenvecs[:, first_k_indices[1:k+1+release]]

    # build matrix b (algorithm line 5)
    b = np.dot(np.dot(d_star, keep_eigenvecs), diag_eigen_values)
    print ('[ESC] Matrix B is derived. size is '+str(b.shape))

    # build matrix q block by block (algorithm line 6~9)
    # since 'row by row' strategy may be inefficient
    if interval > n_instances:
        q = _affinity_vectors_hyperedge(features, selected_features, additional=additional)
    else:
        qs = []
        cur_index = 0
        while True:
            if cur_index + interval > n_instances:
                a_block = _affinity_vectors_hyperedge(features[cur_index:n_instances],
                                                      selected_features, additional=None)
                a_block /= scaler
                i_q = np.dot(a_block, b)
                qs.append(i_q)
                break
            else:
                a_block = _affinity_vectors_hyperedge(features[cur_index:cur_index+interval],
                                                      selected_features, additional=None)
                a_block /= scaler
                i_q = np.dot(a_block, b)
                qs.append(i_q)
                cur_index += interval
        q = np.vstack(qs)

    # conventional methods
    # a = _affinity_vec_methods[method](features[0], selected_features, additional=additional)
    # a /= scaler
    # qs = []
    # qs.append(np.dot(a, b))
    # for i in range(1, n_instances):
    #     a = _affinity_vec_methods[method](features[i], selected_features, additional=additional)
    #     a /= scaler
    #     i_q = np.dot(a, b)
    #     qs.append(i_q)
    # q = np.vstack(qs)
    if verbose:
        print ('[ESC] Matrix Q is derived. size is '+str(q.shape))

    # build matrix d_head (algorithm line 10)
    middle_1 = np.dot(q, np.diag(eigenvalues[first_k_indices[1:k+1+release]]))
    middle_2 = np.dot(q.transpose(), np.ones(n_instances)).transpose()

    d_head = np.dot(middle_1, middle_2)
    d_head = np.asarray(d_head)
    d_head = d_head ** (-0.5)
    d_head = np.asarray(d_head)
    d_head = d_head.reshape(-1)
    if verbose:
        print('[ESC] Matrix D_head is derived. size is ' + str(d_head.shape))

    # # build estimated eigenvector matrix U
    # u = np.empty()
    # for i in range(0, n_instances):
    #     i_u = d_head[i] * q[i]
    #     u = np.hstack([u, i_u])

    # build approximated eigenvectors as u (algorithm line 11)
    # use 'scipy.sparse.dia_matrix' to tackle with sparse matrix multiplication
    diag_d_head = sparse.dia_matrix((d_head, np.array([0])), shape=(n_instances, n_instances))
    u = diag_d_head.dot(q)
    if verbose:
        print ('[ESC] Approximated eigenvector matrix U is derived. size is '+str(u.shape))

    # Orthogonalization
    # p = np.dot(u.transpose(), u)
    # sig, v = LA.eig(p)
    # half_sig = np.diag(sig ** 0.5)
    # bb = np.dot(np.dot(np.dot(half_sig, v.transpose()),
    #                    np.diag(eigenvalues[first_k_indices[1:k+1+release]])),
    #             np.dot(v, half_sig))
    # b_sig, b_v = LA.eig(bb)
    # u = np.dot(u, np.dot(v, np.dot(np.diag(sig ** -0.5), b_v)))
    # print (b_sig)
    # print('[ESC] Orthogonalization Completed.')

    # normalize the matrix U row by row to have unit norm.
    u = preprocessing.normalize(u, norm='l2')

    # utilize k-means to generate clustering result
    kmeans = cluster.KMeans(n_clusters=k, tol=1e-15)
    kmeans.fit(u)

    return kmeans.labels_
