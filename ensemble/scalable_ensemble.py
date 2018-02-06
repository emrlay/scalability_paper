from __future__ import print_function
import time as tt
from build_hypergraph_matrix import *
from sklearn import cluster
import numpy as np
import ensemble.CSPA as CSPA

_VERBOSE_LEVEL = 5
_assign_labels = 'discretize'
_max_iter = 300


def scalable_ensemble_spectral(base_clusterings, n_representatives, class_num,
                               n_km_init, n_spec_init, km_init='k-means++', km_tol=1e-1):
    """
    scalable co-association clustering ensemble using spectral clustering

    Parameters
    ----------
    :param base_clusterings: clusterings to ensemble
    :param n_representatives: number of representative points
    :param class_num: class number of the dataset
    :param n_km_init:
    :param n_spec_init:
    :param km_init:

    Returns
    -------
    :return: ensemble labels
    """
    # get basic information
    n_sols = base_clusterings.shape[0]
    n_samples = base_clusterings.shape[1]
    print ("Ensemble member :" + str(n_sols) + " solutions")
    print ("Sample :" + str(n_samples))

    # build hyper-graph matrix
    t1 = tt.clock()
    adjc = build_hypergraph_adjacency(base_clusterings)
    adjc = adjc.transpose()

    # train k-means model
    km_model = cluster.KMeans(n_clusters=n_representatives, n_init=n_km_init, verbose=_VERBOSE_LEVEL, init=km_init,
                              tol=km_tol, max_iter=20)
    km_model.fit(adjc)
    t2 = tt.clock()
    print ("Representative selection takes:"+str(t2 - t1)+"s")

    # derive reduced similarity matrix
    similarity_matrix = km_model.cluster_centers_.dot(km_model.cluster_centers_.transpose())
    spec_ensembler = cluster.SpectralClustering(n_clusters=class_num, n_init=n_spec_init, affinity='precomputed',
                                                assign_labels=_assign_labels)
    spec_ensembler.fit(similarity_matrix)

    # derive final labels from spectral clustering and mapping vector
    labels = []
    for i in range(0, n_samples):
        labels.append(spec_ensembler.labels_[km_model.labels_[i]])
    labels = np.array(labels)

    return labels


def scalable_ensemble_CSPA(base_clusterings, n_representatives, class_num,
                           n_km_init, km_init='k-means++'):
    """
    scalable co-association clustering ensemble using spectral clustering

    Parameters
    ----------
    :param base_clusterings: clusterings to ensemble
    :param n_representatives: number of representative points
    :param class_num: class number of the dataset
    :param n_km_init:
    :param km_init:

    Returns
    -------
    :return: ensemble labels
    """
    # get basic information
    n_sols = base_clusterings.shape[0]
    n_samples = base_clusterings.shape[1]
    print ("Ensemble member :" + str(n_sols) + " solutions")
    print ("Sample :" + str(n_samples))

    # build hyper-graph matrix
    t1 = tt.clock()
    adjc = build_hypergraph_adjacency(base_clusterings)
    adjc = adjc.transpose()

    # train k-means model
    km_model = cluster.KMeans(n_clusters=n_representatives, n_init=n_km_init, verbose=_VERBOSE_LEVEL, init=km_init, max_iter=1)
    km_model.fit(adjc)
    t2 = tt.clock()
    print ("Representative selection takes:"+str(t2 - t1)+"s")

    # derive reduced similarity matrix
    similarity_matrix = km_model.cluster_centers_.dot(km_model.cluster_centers_.transpose())
    labels_ = CSPA.CSPA_direct(similarity_matrix, class_num)
    print (np.unique(labels_))

    labels = []
    for i in range(0, n_samples):
        labels.append(labels_[km_model.labels_[i]])
    labels = np.array(labels)

    return labels
