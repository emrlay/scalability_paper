import scipy
import numpy as np
import operator


def build_hypergraph_adjacency(cluster_runs):
    """Return the adjacency matrix to a hypergraph, in sparse matrix representation.

    Parameters
    ----------
    cluster_runs : array of shape (n_partitions, n_samples)

    Returns
    -------
    hypergraph_adjacency : compressed sparse row matrix
        Represents the hypergraph associated with an ensemble of partitions,
        each partition corresponding to a row of the array 'cluster_runs'
        provided at input.
    """

    N_runs = cluster_runs.shape[0]

    hypergraph_adjacency = create_membership_matrix(cluster_runs[0])
    for i in xrange(1, N_runs):
        hypergraph_adjacency = scipy.sparse.vstack([hypergraph_adjacency,
                                                    create_membership_matrix(cluster_runs[i])],
                                                   format='csr')

    return hypergraph_adjacency


def create_membership_matrix(cluster_run):
    """For a label vector represented by cluster_run, constructs the binary
        membership indicator matrix. Such matrices, when concatenated, contribute
        to the adjacency matrix for a hypergraph representation of an
        ensemble of clusterings.

    Parameters
    ----------
    cluster_run : array of shape (n_partitions, n_samples)

    Returns
    -------
    An adjacnecy matrix in compressed sparse row form.
    """

    cluster_run = np.asanyarray(cluster_run)

    if reduce(operator.mul, cluster_run.shape, 1) != max(cluster_run.shape):
        raise ValueError("\nERROR: Cluster_Ensembles: create_membership_matrix: "
                         "problem in dimensions of the cluster label vector "
                         "under consideration.")
    else:
        cluster_run = cluster_run.reshape(cluster_run.size)

        cluster_ids = np.unique(np.compress(np.isfinite(cluster_run), cluster_run))

        indices = np.empty(0, dtype=np.int32)
        indptr = np.zeros(1, dtype=np.int32)

        for elt in cluster_ids:
            indices = np.append(indices, np.where(cluster_run == elt)[0])
            indptr = np.append(indptr, indices.size)

        data = np.ones(indices.size, dtype=int)

        return scipy.sparse.csr_matrix((data, indices, indptr), shape=(cluster_ids.size, cluster_run.size))
