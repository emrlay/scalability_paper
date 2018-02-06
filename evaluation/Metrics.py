import numpy as np
from sklearn.metrics.cluster import entropy
from sklearn.metrics.cluster import contingency_matrix
from sklearn import metrics
import itertools


def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays (internal use only)
       Copy from sklearn.metrics.cluster.supervised since it is not defined at the '__init__'
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred


def normalized_max_mutual_info_score(labels_true, labels_pred):
    """
    A variant version of NMI that is given as:
    NMI_max = MI(U, V) / max{ H(U), H(V) }
    based on 'adjusted mutual info score' in sklearn

    Parameters
    ----------
    :param labels_true: labels of clustering 1 (as a 1-dimensional ndarray)
    :param labels_pred: labels of clustering 2 (as a 1-dimensional ndarray)
    :return: diversity between these two clusterings as a float value

    Returns
    -------
    :return: NMI-max between these two clusterings as a float value

    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split.
    # This is a perfect match hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0):
        return 1.0
    contingency = contingency_matrix(labels_true, labels_pred)
    print contingency
    contingency = np.array(contingency, dtype='float')
    # Calculate the MI for the two clusterings
    mi = metrics.mutual_info_score(labels_true, labels_pred,
                           contingency=contingency)
    # Calculate the expected value for the mutual information
    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    nmi_max = mi / max(h_true, h_pred)
    return nmi_max


def normalized_mutual_info_score(labels_true, labels_pred):
    """
    A variant version of NMI that is given as:
    NMI_max = MI(U, V) / max{ H(U), H(V) }
    based on 'adjusted mutual info score' in sklearn

    Parameters
    ----------
    :param labels_true: labels of clustering 1 (as a 1-dimensional ndarray)
    :param labels_pred: labels of clustering 2 (as a 1-dimensional ndarray)
    :return: diversity between these two clusterings as a float value

    Returns
    -------
    :return: NMI-max between these two clusterings as a float value

    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split.
    # This is a perfect match hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0):
        return 1.0
    contingency = contingency_matrix(labels_true, labels_pred)
    contingency = np.array(contingency, dtype='float')
    # Calculate the MI for the two clusterings
    mi = metrics.mutual_info_score(labels_true, labels_pred,
                           contingency=contingency)
    # Calculate the expected value for the mutual information
    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    nmi_max = mi / (h_true + h_pred)
    return nmi_max


def find_best_match(label_true, label_pred):
    """
    find best match between true label and predicted label

    Parameters
    ----------
    :param label_true: true labels as 1d array
    :param label_pred: predicted labels as 1d array

    Returns
    -------
    :return: match as a dictionary
    """
    if len(label_true) != len(label_pred):
        raise ValueError("[FIND_BEST_MATCH] length of true labels and predicted labels should be the same")
    best_match = dict(zip(np.unique(label_pred), [-1]*len(label_true)))
    real_class = np.unique(label_true)
    predicted_cluster = np.unique(label_pred)
    match_num_matrix = []
    for clu in predicted_cluster:
        match_num = [] * len(real_class)
        for cla in real_class:
            overlap_num = np.logical_and(label_true == cla, label_pred == clu).astype(int).sum()
            match_num.append(overlap_num)
        match_num_matrix.append(match_num)
    match_num_matrix = np.array(match_num_matrix)
    for cla in real_class:
        predicted_cluster_rank = np.argsort(-match_num_matrix[:, cla])
        for clu in predicted_cluster_rank:
            if best_match[clu] == -1:
                best_match[clu] = cla
                break
    return best_match, match_num_matrix


def precision(label_true, label_pred):
    """
    clustering precision between true labels and predicted labels
    based on find_best_match

    Parameters
    ----------
    :param label_true: true labels as 1d array
    :param label_pred: predicted labels as 1d array

    Returns
    -------
    :return: precision as a double value
    """
    best_match, _ = find_best_match(label_true, label_pred)
    predicted_clusters = np.unique(label_pred)
    cur_num = 0
    n_samples = len(label_true)
    for clu in predicted_clusters:
        cur_num += np.logical_and(label_pred == clu, label_true == best_match[clu]).astype(int).sum()
    return float(cur_num) / n_samples

# def _permutation(len):
    # best_match