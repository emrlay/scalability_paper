import numpy as np
import ensemble.build_hypergraph_matrix  as bhm
import evaluation.Metrics as Metrics
from sklearn import cluster
import ensemble.scalable_ensemble as se
from sklearn import preprocessing
import scipy.io as sio
import ensemble.spectral_ensemble as spec
import ensemble.CSPA as CSPA
import scipy.sparse
from sklearn import metrics
import time
import gc


def load_banknote(normalized=True):
    data = []
    target = []
    fr = open('data/data_banknote_authentication.txt')
    for line in fr.readlines():
        cur = line.strip().split(',')
        target.append(int(cur[-1]))
        cur = cur[:-1]
        flt_line = map(float, cur)
        data.append(flt_line)

    data = np.array(data)
    target = np.array(target)
    if normalized:
        min_max_scaler = preprocessing.MinMaxScaler()
        data = min_max_scaler.fit_transform(data)

    return data, target


def load_digit():
    data = []
    target = []
    fr = open('data/optdigits.tra')
    for line in fr.readlines():
        cur = line.strip().split(',')
        target.append(int(cur[-1]))
        cur = cur[:-1]
        flt_line = map(float, cur)
        data.append(flt_line)

    data = np.array(data)
    target = np.array(target)

    return data, target


def loadIsolet():
    """
            load isolet dataset
        :return:
        """
    print ('Loading ISOLET...')
    dataSet = []
    target = []
    fr = open('data/isolet5.data')

    for line in fr.readlines():
        curLine = line.strip().split(',')
        target.append(int((curLine[-1]).replace('.', '')))
        curLine = curLine[:-1]
        # curLine.remove(curLine[-1])
        fltLine = map(float, curLine)
        dataSet.append(fltLine)

    return np.array(dataSet), np.array(target) - 1


def load_musk_2_data(normalized=True):
    fr = open('data/Musk-2.data')
    count = 0
    feature_vectors = []
    target = []
    for line in fr.readlines():
        count += 1
        line_elements = line.strip().split(',')
        feature = line_elements[2:-1]
        feature = map(float, feature)
        feature_vectors.append(feature)
        target.append(0 if line_elements[-1] == '0.' else 1)
        # print str(count) + " -- " + str(len(line_elements)) + "  " + line_elements[-1]
    target = np.array(target)
    feature_vectors = np.array(feature_vectors)
    print feature_vectors.shape
    if normalized:
        data_normed = preprocessing.normalize(feature_vectors, norm='l2')
        return data_normed, target
        # min_max_scaler = preprocessing.MinMaxScaler()
        # feature_vectors = min_max_scaler.fit_transform(feature_vectors)
    # print np.sum(target)
    # print target
    return feature_vectors, target


def load_sat(normalized=True):
    fr = open('UCI Data/SAT/sat.data')
    count = 0
    feature_vectors = []
    target = []
    for line in fr.readlines():
        count += 1
        line_elements = line.strip().split(' ')
        feature = line_elements[0:-1]
        feature = map(float, feature)
        feature_vectors.append(feature)
        target.append(5 if line_elements[-1] == '7' else (int(line_elements[-1]) - 1))
        print str(count) + " -- " + str(len(line_elements)) + "  " + line_elements[-1]
    target = np.array(target)
    feature_vectors = np.array(feature_vectors)
    print feature_vectors.shape
    if normalized:
        data_normed = preprocessing.normalize(feature_vectors, norm='l2')
        return data_normed, target
        # min_max_scaler = preprocessing.MinMaxScaler()
        # feature_vectors = min_max_scaler.fit_transform(feature_vectors)
    # print np.sum(target)
    # print target
    return feature_vectors, target


def load_mnist_full():
    data = sio.loadmat('data/Orig.mat')
    fea = data['fea']
    fea = fea.astype(np.float64)
    labels = data['gnd'] - 1
    labels = labels.flatten()
    labels[labels == 255] = 9
    return fea, labels


def load_skin(normalized=True):
    dataset = []
    target = []
    fr = open('data/Skin_NonSkin.txt')

    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        target.append(int(cur_line[-1]))
        cur_line = cur_line[:-1]
        # curLine.remove(curLine[-1])
        flt_line = map(float, cur_line)
        dataset.append(flt_line)
    dataset = np.array(dataset)
    target = np.array(target)
    target = target - 1
    if normalized:
        data_normed = preprocessing.normalize(dataset, norm='l2')
        return data_normed, target
        # min_max_scaler = preprocessing.MinMaxScaler()
        # dataset = min_max_scaler.fit_transform(dataset)

    return dataset, target


def load_usps():
    data = sio.loadmat('data/USPS.mat')
    fea = data['fea']
    fea = fea.astype(np.float64)
    labels = data['gnd'] - 1
    labels = labels.flatten()
    # labels[labels == 255] = 9
    return fea, labels


def load_covtype():
    data = np.loadtxt('data/covtype.data', delimiter=',')
    print data.shape
    targets = data[:, -1].flatten() - 1
    print np.unique(targets)
    print targets.shape
    data = data[:, :-1]
    print data.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(data)
    return X_train_minmax, targets


def load_coil20():
    data = sio.loadmat('data/COIL20.mat')
    fea = data['fea']
    fea = fea.astype(np.float64)
    labels = data['gnd'] - 1
    labels = labels.flatten()
    return fea, labels


def load_mnist_4000():
    data = sio.loadmat('data/2k2k.mat')
    fea = data['fea']
    fea = fea.astype(np.float64)
    labels = data['gnd'] - 1
    labels = labels.flatten()
    labels[labels == 255] = 9
    return fea, labels

if __name__ == "__main__":
    """
    HEPMASS
    """
    # data = np.loadtxt('E:/select1m.csv', delimiter=',')
    # t = data[:, 0].astype(np.int)
    # d = data[:, 1:]
    # gc.collect()
    # lib = np.loadtxt('library/HEPMASS_5-10_0.7_0.7_50_FSRSNC_pure.res', delimiter=',')
    # print 'Load Feature completed.!'
    # label = se.scalable_ensemble_spectral(lib, 280, 2, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # label = se.scalable_ensemble_spectral(lib, 360, 2, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # label = se.scalable_ensemble_spectral(lib, 400, 2, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print 'all completed.'


    """
    USPS
    """
    # d, t = load_usps()
    # km = cluster.KMeans(n_clusters=10, n_init=1)
    # km.fit(d)
    # label = km.labels_
    # lib = np.loadtxt('library/USPS_50-100_0.5_0.3_160_FSRSNC_pure.res', delimiter=',')
    # adjc = bhm.build_hypergraph_adjacency(lib)
    # s = scipy.sparse.csr_matrix.dot(adjc.transpose().tocsr(), adjc)
    # s = np.squeeze(np.asarray(s.todense()))
    # label = CSPA.CSPA_direct(s, 10)
    # label = spec.spectral_ensemble(lib, 10, 10)
    # label = se.scalable_ensemble_CSPA(lib, 100, 10, 1)
    # label = se.scalable_ensemble_spectral(lib, 160, 10, 1, 10)
    # print metrics.normalized_mutual_info_score(t, label)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)

    """
    SKIN
    """
    # d, t = load_skin()
    # print np.unique(t)
    # km = cluster.KMeans(n_clusters=2, n_init=1)
    # km.fit(d)
    # label = km.labels_
    # print t.dtype
    # t = t.astype(np.int32)
    # t = t - 1
    # lib = np.loadtxt('library/SKIN_10-20_0.5_1.0_160_FSRSNC_pure.res', delimiter=',')
    # label = se.scalable_ensemble_CSPA(lib, 200, 2, 1)
    # label = se.scalable_ensemble_spectral(lib, 200, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '========================================================'
    # label = se.scalable_ensemble_CSPA(lib, 250, 2, 1)
    # label = se.scalable_ensemble_spectral(lib, 200, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)

    # CSPA
    # print '10-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 11, 2, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '10-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 11, 2, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '20-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 20, 2, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '20-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 20, 2, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '40-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 40, 2, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '40-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 40, 2, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '80-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 80, 2, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '80-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 80, 2, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '160-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 160, 2, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '160-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 160, 2, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '240-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 240, 2, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '240-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 240, 2, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '320-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 320, 2, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '320-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 320, 2, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '400-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 400, 2, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '400-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 400, 2, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)

    # spec
    # print '10-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 11, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '10-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 11, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '20-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 20, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '20-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 20, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '40-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 40, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '40-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 40, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '80-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 80, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '80-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 80, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '160-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 160, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '160-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 160, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '240-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 240, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '240-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 240, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '320-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 320, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '320-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 320, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '400-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 400, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '400-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 400, 2, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)

    """
    MNIST_FULL
    """
    # d, t = load_mnist_full()
    # km = cluster.KMeans(n_clusters=10, n_init=1)
    # km.fit(d)
    # label = km.labels_
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # lib = np.loadtxt('library/MNIST_FULL_10-100_0.7_0.5_160_FSRSNC_pure.res', delimiter=',')
    # print '10-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 11, 10, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '10-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 11, 10, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '20-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 20, 10, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '20-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 20, 10, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '40-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 40, 10, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '40-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 40, 10, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '80-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 80, 10, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '80-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 80, 10, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '160-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 160, 10, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '160-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 160, 10, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '240-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 240, 10, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '240-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 240, 10, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '320-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 320, 10, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '320-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 320, 10, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '400-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 400, 10, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '400-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 400, 10, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)

    # spec
    # print '10-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 11, 10, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '10-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 11, 10, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '20-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 20, 10, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '20-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 20, 10, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '40-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 40, 10, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '40-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 40, 10, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '80-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 80, 10, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '80-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 80, 10, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '160-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 160, 10, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '160-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 160, 10, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '240-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 240, 10, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '240-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 240, 10, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '320-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 320, 10, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '320-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 320, 10, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '400-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 400, 10, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '400-SSC==================================================================================='
    # label = se.scalable_ensemble_spectral(lib, 400, 10, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)


    """
    ISOLET-5
    """
    d, t = loadIsolet()
    lib = np.loadtxt('library/ISOLET_26-46_0.5_0.3_160_FSRSNC.res', delimiter=',')
    lib = lib[0:-5]
    # km = cluster.KMeans(n_clusters=26)
    # km.fit(d)
    # label = km.labels_
    # label = CSPA.CSPA(lib, 26)
    # label = spec.spectral_ensemble(lib, 26, 10)
    label = se.scalable_ensemble_spectral(lib, 104, 26, 1, 10)
    # label = se.scalable_ensemble_CSPA(lib, 500, 26, 1)
    print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)

    # print lib.shape
    # adjc = bhm.build_hypergraph_adjacency(lib)
    # adjc = adjc.transpose()
    # aff_mat = adjc.dot(adjc.transpose())
    # spec = cluster.SpectralClustering(n_clusters=26, affinity='precomputed', n_init=10)
    # spec.fit(aff_mat)
    # print Metrics.normalized_max_mutual_info_score(t, spec.labels_)

    """
    """
    # d, t = load_covtype()
    # km = cluster.KMeans(n_clusters=7)
    # km.fit(d)
    # label = km.labels_
    # lib = np.loadtxt('library/covtype_labels.txt', delimiter=',')
    # adjc = bhm.build_hypergraph_adjacency(lib)
    # print adjc.shape
    # s = scipy.sparse.csr_matrix.dot(adjc.transpose().tocsr(), adjc)
    # CSPA
    # print '10-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 11, 7, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '10-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 11, 7, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '20-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 20, 7, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '20-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 20, 7, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '40-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 40, 7, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '40-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 40, 7, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '80-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 80, 7, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '80-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 80, 7, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '160-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 160, 7, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '160-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 160, 7, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '240-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 240, 7, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '240-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 240, 7, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '320-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 320, 7, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '320-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 320, 7, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '400-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 400, 7, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '400-CSPA==================================================================================='
    # label = se.scalable_ensemble_CSPA(lib, 400, 7, 1)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)


    # label = se.scalable_ensemble_CSPA(lib, 300, 7, 1, km_init='random')
    # print np.unique(label)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print metrics.normalized_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '10============================================================================'
    # label = se.scalable_ensemble_spectral(lib, 10, 7, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '10============================================================================'
    # label = se.scalable_ensemble_spectral(lib, 10, 7, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '20============================================================================'
    # label = se.scalable_ensemble_spectral(lib, 20, 7, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '20============================================================================'
    # label = se.scalable_ensemble_spectral(lib, 20, 7, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '40============================================================================'
    # label = se.scalable_ensemble_spectral(lib, 40, 7, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '40============================================================================'
    # label = se.scalable_ensemble_spectral(lib, 40, 7, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '80============================================================================'
    # label = se.scalable_ensemble_spectral(lib, 80, 7, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '80============================================================================'
    # label = se.scalable_ensemble_spectral(lib, 80, 7, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '140============================================================================'
    # label = se.scalable_ensemble_spectral(lib, 140, 7, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '140============================================================================'
    # label = se.scalable_ensemble_spectral(lib, 140, 7, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '200============================================================================'
    # label = se.scalable_ensemble_spectral(lib, 200, 7, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '200============================================================================'
    # label = se.scalable_ensemble_spectral(lib, 200, 7, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '280============================================================================'
    # label = se.scalable_ensemble_spectral(lib, 280, 7, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '280============================================================================'
    # label = se.scalable_ensemble_spectral(lib, 280, 7, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '360============================================================================'
    # label = se.scalable_ensemble_spectral(lib, 360, 7, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print '360============================================================================'
    # label = se.scalable_ensemble_spectral(lib, 360, 7, 1, 10, km_init='random')
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    # print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    """
    COIL20
    """
    # d, t = load_coil20()
    # lib = np.loadtxt('library/COIL20_20-200_0.5_0.3_160_FSRSNC.res', delimiter=',')
    # lib = lib[0:-5]
    # label = CSPA.CSPA(lib, 20)
    # label = se.scalable_ensemble_CSPA(lib, 750, 20, 1)
    # label = se.scalable_ensemble_spectral(lib, 1000, 20, 1, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)

    # d, t = load_digit()
    # lib = np.loadtxt('library/OPTDIGITS_10-100_0.5_0.3_160_FSRSNC_pure.res', delimiter=',')
    # label = se.scalable_ensemble_CSPA(lib, 400, 10, 1)
    # print d.shape
    # print t.shape
    # km = cluster.KMeans(n_clusters=10, n_init=1, init='random')
    # km.fit(d)
    # label = km.labels_
    # label = CSPA.CSPA(lib, 10)
    # label = spec.spectral_ensemble(lib, 10, 10)
    # print np.unique(t)
    # print np.unique(km.labels_)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)

    """
    MNIST4000
    """
    # d, t = load_mnist_4000()
    # lib = np.loadtxt('library/MNIST4000_10-100_0.5_0.8_160_FSRSNC.res', delimiter=',')
    # lib = lib[0:-5]
    # label = se.scalable_ensemble_CSPA(lib, 240, 10, 1)
    # km = cluster.KMeans(n_clusters=10, n_init=1)
    # km.fit(d)
    # label = km.labels_
    # label = spec.spectral_ensemble(lib, 10, 10)
    # label = CSPA.CSPA(lib, 10)
    # print Metrics.normalized_max_mutual_info_score(t, label)
    # print Metrics.precision(t, label)
    pass
