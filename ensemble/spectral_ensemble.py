from __future__ import print_function
from sklearn import cluster
from build_hypergraph_matrix import *
import numpy as np
import time as tt

_assign_labels = 'discretize'


def spectral_ensemble(base_clusterings, class_num, n_spec_init):
    # get basic information
    n_sols = base_clusterings.shape[0]
    n_samples = base_clusterings.shape[1]
    print("Ensemble member :" + str(n_sols) + " solutions")
    print("Sample :" + str(n_samples))

    # build hyper-graph matrix
    t1 = tt.clock()
    adjc = build_hypergraph_adjacency(base_clusterings)
    adjc = adjc.transpose()

    att_mat = adjc.dot(adjc.transpose())
    att_mat = np.squeeze(np.asarray(att_mat.todense()))

    spec_ensembler = cluster.SpectralClustering(n_clusters=class_num, n_init=n_spec_init, affinity='precomputed',
                                                assign_labels=_assign_labels)

    spec_ensembler.fit(att_mat)

    return spec_ensembler.labels_
