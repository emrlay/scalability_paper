from __future__ import print_function
import time as tt
from build_hypergraph_matrix import *
from sklearn import cluster
import numpy as np


def scalable_ensemble_spectral(base_clusterings, n_representatives, class_num, n_km_init, n_spec_init, km_init):
    """
    scalable co-association clustering ensemble using spectral clustering

    :param base_clusterings:
    :param n_representatives:
    :param class_num:
    :param n_km_init:
    :param n_spec_init:
    :param km_init:
    :return:
    """
    # get basic information
    n_sols = base_clusterings.shape[0]
    n_samples = base_clusterings.shape[1]
    print ("Ensemble member :" + str(n_sols) + " solutions")
    print ("Sample :" + str(n_samples))

    # build hyper-graph matrix
    adjc = build_hypergraph_adjacency(base_clusterings)
    adjc = adjc.transpose()

    # train k-means model
    km_model = cluster.KMeans(n_clusters=n_representatives, n_init=n_km_init, verbose=5, init=km_init)
    t1 = tt.clock()
    km_model.fit(adjc)
    t2 = tt.clock()
    print ("KMeans model, training takes:"+str(t2 - t1)+"s")

    # derive reduced similarity matrix
    similarity_matrix = km_model.cluster_centers_.dot(km_model.cluster_centers_.transpose())
    spec_ensembler = cluster.SpectralClustering(n_clusters=class_num, n_init=n_spec_init, affinity='precomputed')
    spec_ensembler.fit(similarity_matrix)

    # derive final labels from spectral clustering and mapping vector
    labels = []
    for i in range(0, n_samples):
        labels.append(spec_ensembler.labels_[km_model.labels_[i]])
    labels = np.array(labels)

    return labels
