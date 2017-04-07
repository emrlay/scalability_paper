import scipy.sparse
import gc
from build_hypergraph_matrix import *
import tables
import psutil
import sys
import numbers
import os
import subprocess
import pkg_resources

hdf5_file_name = 'aaa'


def memory():
    """Determine memory specifications of the machine.

    Returns
    -------
    mem_info : dictonary
        Holds the current values for the total, free and used memory of the system.
    """

    mem_info = {}
    memory_stat = psutil.virtual_memory()
    mem_info['total'] = int(memory_stat.total / 1024)
    mem_info['free'] = int(memory_stat.available / 1024)

    return mem_info


def one_to_max(array_in):
    """Alter a vector of cluster labels to a dense mapping.
        Given that this function is herein always called after passing
        a vector to the function checkcl, one_to_max relies on the assumption
        that cluster_run does not contain any NaN entries.

    Parameters
    ----------
    array_in : a list or one-dimensional array
        The list of cluster IDs to be processed.

    Returns
    -------
    result : one-dimensional array
        A massaged version of the input vector of cluster identities.
    """

    x = np.asanyarray(array_in)
    N_in = x.size
    array_in = x.reshape(N_in)

    sorted_array = np.sort(array_in)
    sorting_indices = np.argsort(array_in)

    last = np.nan
    current_index = -1
    for i in xrange(N_in):
        if last != sorted_array[i] or np.isnan(last):
            last = sorted_array[i]
            current_index += 1

        sorted_array[i] = current_index

    result = np.empty(N_in, dtype=int)
    result[sorting_indices] = sorted_array

    return result


def wgraph(hdf5_file_name, w=None, method=0, HGPA_weighted=False):
    """Write a graph file in a format apposite to later use by METIS or HMETIS.

    Parameters
    ----------
    hdf5_file_name : file handle or string

    w : list or array, optional (default = None)

    method : int, optional (default = 0)

    HGPA_weighted :

    Returns
    -------
    file_name : string
    """

    print('\n#')

    if method == 0:
        fileh = tables.open_file(hdf5_file_name, 'r+')
        e_mat = fileh.root.consensus_group.similarities_CSPA
        file_name = 'wgraph_CSPA'
    elif method == 1:
        raise ValueError("\nERROR: Cluster_Ensembles: wgraph: "
                         "invalid code for choice of method; ")
    elif method in {2, 3}:
        raise ValueError("\nERROR: Cluster_Ensembles: wgraph: "
                         "invalid code for choice of method; ")
    else:
        raise ValueError("\nERROR: Cluster_Ensembles: wgraph: "
                         "invalid code for choice of method; ")

    if w is None:
        w = []

    N_rows = e_mat.shape[0]
    N_cols = e_mat.shape[1]

    if method in {0, 1}:
        diag_ind = np.diag_indices(N_rows)
        e_mat[diag_ind] = 0

    if method == 1:
        scale_factor = 100.0
        w_sum_before = np.sum(w)
        w *= scale_factor
        w = np.rint(w)

    with open(file_name, 'w') as file:
        print("INFO: Cluster_Ensembles: wgraph: writing {}.".format(file_name))

        if method == 0:
            sz = float(np.sum(e_mat[:] > 0)) / 2
            if int(sz) == 0:
                return 'DO_NOT_PROCESS'
            else:
                # the first line of METIS input format is: node_num, edge_num, is_weighted
                # edge_num eqs half the amount of elements greater than 0 in similarity matrices
                file.write('{} {} 1\n'.format(N_rows, int(sz)))
        elif method == 1:
            chunks_size = get_chunk_size(N_cols, 2)
            N_chunks, remainder = divmod(N_rows, chunks_size)
            if N_chunks == 0:
                sz = float(np.sum(e_mat[:] > 0)) / 2
            else:
                sz = 0
                for i in xrange(N_chunks):
                    M = e_mat[i * chunks_size:(i + 1) * chunks_size]
                    sz += float(np.sum(M > 0))
                if remainder != 0:
                    M = e_mat[N_chunks * chunks_size:N_rows]
                    sz += float(np.sum(M > 0))
                sz = float(sz) / 2
            file.write('{} {} 11\n'.format(N_rows, int(sz)))
        else:
            file.write('{} {} 1\n'.format(N_cols, N_rows))

        if method in {0, 1}:
            chunks_size = get_chunk_size(N_cols, 2)
            for i in xrange(0, N_rows, chunks_size):
                # read a chunk of similarity matrix from the disk
                M = e_mat[i:min(i + chunks_size, N_rows)]

                for j in xrange(M.shape[0]):
                    # get indices of elements that greater than 0 (connected vertices)
                    edges = np.where(M[j] > 0)[0]
                    # get weights
                    weights = M[j, edges]

                    if method == 0:
                        # each line lists connected vertices as format (vertex_1 weight_1  ... vertex_n weight_n)
                        interlaced = np.zeros(2 * edges.size, dtype=int)
                        # METIS and hMETIS have vertices numbering starting from 1:
                        interlaced[::2] = edges + 1
                        interlaced[1::2] = weights
                    else:
                        interlaced = np.zeros(1 + 2 * edges.size, dtype=int)
                        interlaced[0] = w[i + j]
                        # METIS and hMETIS have vertices numbering starting from 1:
                        interlaced[1::2] = edges + 1
                        interlaced[2::2] = weights

                    for elt in interlaced:
                        file.write('{} '.format(int(elt)))
                    file.write('\n')
        else:
            print("INFO: Cluster_Ensembles: wgraph: {N_rows} vertices and {N_cols} "
                  "non-zero hyper-edges.".format(**locals()))

            chunks_size = get_chunk_size(N_rows, 2)

            scaler = 1000
            for i in xrange(0, N_cols, chunks_size):
                M = np.asarray(e_mat[:, i:min(i + chunks_size, N_cols)].todense())
                for j in xrange(M.shape[1]):
                    edges = np.where(M[:, j] > 0)[0]
                    if method == 2:
                        if HGPA_weighted:
                            weight = np.array(M[:, j].sum() * scaler, dtype=int)
                        else:
                            weight = np.array(M[:, j].sum(), dtype=int)
                        if int(M[:, j].sum() * scaler) == 0:
                            print (M[:, j].sum())
                    else:
                        weight = w[i + j]
                    # METIS and hMETIS require vertices numbering starting from 1:
                    interlaced = np.append(weight, edges + 1)

                    for elt in interlaced:
                        file.write('{} '.format(int(elt)))
                    file.write('\n')

    if method in {0, 1}:
        fileh.remove_node(fileh.root.consensus_group, e_mat.name)

    fileh.close()

    print('#')

    return file_name


def sgraph(N_clusters_max, file_name):
    """Runs METIS or hMETIS and returns the labels found by those
        (hyper-)graph partitioning algorithms.

    Parameters
    ----------
    N_clusters_max : int

    file_name : string

    Returns
    -------
    labels : array of shape (n_samples,)
        A vector of labels denoting the cluster to which each sample has been assigned
        as a result of any of three approximation algorithms for consensus clustering
        (either of CSPA, HGPA or MCLA).
    """

    if file_name == 'DO_NOT_PROCESS':
        return []

    print('\n#')

    k = str(N_clusters_max)
    out_name = file_name + '.part.' + k
    if file_name == 'wgraph_HGPA':
        print("INFO: Cluster_Ensembles: sgraph: "
              "calling shmetis for hypergraph partitioning.")

        if sys.platform.startswith('linux'):
            shmetis_path = pkg_resources.resource_filename(__name__,
                                                           'Hypergraph_Partitioning/hmetis-1.5-linux/shmetis')
        elif sys.platform.startswith('darwin'):
            shmetis_path = pkg_resources.resource_filename(__name__,
                                                           'Hypergraph_Partitioning/hmetis-1.5-osx-i686/shmetis')
        else:
            shmetis_path = pkg_resources.resource_filename(__name__,
                                                           'Hypergraph_Partitioning/windows/shmetis')

        args = "{0} ./".format(shmetis_path) + file_name + " " + k + " 15"
        subprocess.call(args, shell=True)
    elif file_name == 'wgraph_CSPA' or file_name == 'wgraph_MCLA':
        print("INFO: Cluster_Ensembles: sgraph: "
              "calling gpmetis for graph partitioning.")
        args = "gpmetis ./" + file_name + " " + k
        subprocess.call(args, shell=True)
    else:
        raise NameError("ERROR: Cluster_Ensembles: sgraph: {} is not an acceptable "
                        "file-name.".format(file_name))

    labels = np.empty(0, dtype=int)
    with open(out_name, 'r') as file:
        print("INFO: Cluster_Ensembles: sgraph: (hyper)-graph partitioning completed; "
              "loading {}".format(out_name))
        labels = np.loadtxt(out_name, dtype=int)
        labels = labels.reshape(labels.size)
    labels = one_to_max(labels)

    # subprocess.call(['rm', out_name])
    os.remove(out_name)

    print('#')

    return labels



def get_chunk_size(N, n):
    """Given a two-dimensional array with a dimension of size 'N',
        determine the number of rows or columns that can fit into memory.

    Parameters
    ----------
    N : int
        The size of one of the dimensions of a two-dimensional array.

    n : int
        The number of arrays of size 'N' times 'chunk_size' that can fit in memory.

    Returns
    -------
    chunk_size : int
        The size of the dimension orthogonal to the one of size 'N'.
    """

    mem_free = memory()['free']
    if mem_free > 60000000:
        chunk_size = int(((mem_free - 10000000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 40000000:
        chunk_size = int(((mem_free - 7000000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 14000000:
        chunk_size = int(((mem_free - 2000000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 8000000:
        chunk_size = int(((mem_free - 1400000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 2000000:
        chunk_size = int(((mem_free - 900000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 1000000:
        chunk_size = int(((mem_free - 400000) * 1000) / (4 * n * N))
        return chunk_size
    else:
        print("\nERROR: Cluster_Ensembles: get_chunk_size: "
              "this machine does not have enough free memory resources "
              "to perform ensemble clustering.\n")
        sys.exit(1)


def get_compression_filter(byte_counts):
    """Determine whether or not to use a compression on the array stored in
        a hierarchical data format, and which compression library to use to that purpose.
        Compression reduces the HDF5 file size and also helps improving I/O efficiency
        for large datasets.

    Parameters
    ----------
    byte_counts : int

    Returns
    -------
    FILTERS : instance of the tables.Filters class
    """

    assert isinstance(byte_counts, numbers.Integral) and byte_counts > 0

    if 2 * byte_counts > 1000 * memory()['free']:
        try:
            FILTERS = tables.Filters(complevel=5, complib='blosc',
                                     shuffle=True, least_significant_digit=6)
        except tables.FiltersWarning:
            FILTERS = tables.Filters(complevel=5, complib='lzo',
                                     shuffle=True, least_significant_digit=6)
    else:
        FILTERS = None

    return FILTERS


def CSPA_direct(att_mat, class_num):

    fileh = tables.open_file(hdf5_file_name, 'w')
    fileh.create_group(fileh.root, 'consensus_group')
    fileh.close()

    print('*****')
    print("INFO: Cluster_Ensembles: CSPA: consensus clustering using CSPA.")
    N_samples = att_mat.shape[0]

    s = att_mat

    gc.collect()

    e_sum_before = s.sum()
    sum_after = 100000000.0
    scale_factor = sum_after / float(e_sum_before)

    with tables.open_file(hdf5_file_name, 'r+') as fileh:
        atom = tables.Float32Atom()
        FILTERS = get_compression_filter(4 * (N_samples ** 2))

        S = fileh.create_carray(fileh.root.consensus_group, 'similarities_CSPA', atom,
                                (N_samples, N_samples), "Matrix of similarities arising "
                                                        "in Cluster-based Similarity Partitioning",
                                filters=FILTERS)

        expr = tables.Expr("s * scale_factor")
        expr.set_output(S)
        expr.eval()

        chunks_size = get_chunk_size(N_samples, 3)
        for i in xrange(0, N_samples, chunks_size):
            tmp = S[i:min(i + chunks_size, N_samples)]
            S[i:min(i + chunks_size, N_samples)] = np.rint(tmp)

    return metis(hdf5_file_name, class_num)


def CSPA(base_clusterings, class_num):

    fileh = tables.open_file(hdf5_file_name, 'w')
    fileh.create_group(fileh.root, 'consensus_group')
    fileh.close()

    print('*****')
    print("INFO: Cluster_Ensembles: CSPA: consensus clustering using CSPA.")

    N_runs = base_clusterings.shape[0]
    N_samples = base_clusterings.shape[1]
    if N_samples > 20000:
        raise ValueError("\nERROR: Cluster_Ensembles: CSPA: cannot efficiently "
                         "deal with too large a number of cells.")

    hypergraph_adjacency = build_hypergraph_adjacency(base_clusterings)

    s = scipy.sparse.csr_matrix.dot(hypergraph_adjacency.transpose().tocsr(), hypergraph_adjacency)

    s = np.squeeze(np.asarray(s.todense()))

    del hypergraph_adjacency
    gc.collect()

    e_sum_before = s.sum()
    sum_after = 100000000.0
    scale_factor = sum_after / float(e_sum_before)

    with tables.open_file(hdf5_file_name, 'r+') as fileh:
        atom = tables.Float32Atom()
        FILTERS = get_compression_filter(4 * (N_samples ** 2))

        S = fileh.create_carray(fileh.root.consensus_group, 'similarities_CSPA', atom,
                                (N_samples, N_samples), "Matrix of similarities arising "
                                                        "in Cluster-based Similarity Partitioning",
                                filters=FILTERS)

        expr = tables.Expr("s * scale_factor")
        expr.set_output(S)
        expr.eval()

        chunks_size = get_chunk_size(N_samples, 3)
        for i in xrange(0, N_samples, chunks_size):
            tmp = S[i:min(i + chunks_size, N_samples)]
            S[i:min(i + chunks_size, N_samples)] = np.rint(tmp)

    return metis(hdf5_file_name, class_num)


def metis(hdf5_file_name, N_clusters_max):
    """METIS algorithm by Karypis and Kumar. Partitions the induced similarity graph
        passed by CSPA.

    Parameters
    ----------
    hdf5_file_name : string or file handle

    N_clusters_max : int

    Returns
    -------
    labels : array of shape (n_samples,)
        A vector of labels denoting the cluster to which each sample has been assigned
        as a result of the CSPA heuristics for consensus clustering.

    Reference
    ---------
    G. Karypis and V. Kumar, "A Fast and High Quality Multilevel Scheme for
    Partitioning Irregular Graphs"
    In: SIAM Journal on Scientific Computing, Vol. 20, No. 1, pp. 359-392, 1999.
    """

    file_name = wgraph(hdf5_file_name)
    labels = sgraph(N_clusters_max, file_name)

    # subprocess.call(['rm', file_name])
    os.remove(file_name)

    return labels


