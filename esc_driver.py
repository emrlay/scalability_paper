import ensemble.tande as esc
from ensemble.build_hypergraph_matrix import *
import evaluation.Metrics as Metrics
# from sklearn import metrics
import scipy.io as sio
from sklearn import preprocessing
import gc


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


def load_usps():
    data = sio.loadmat('data/USPS.mat')
    fea = data['fea']
    fea = fea.astype(np.float64)
    labels = data['gnd'] - 1
    labels = labels.flatten()
    # labels[labels == 255] = 9
    return fea, labels


def load_mnist_4000():
    data = sio.loadmat('data/2k2k.mat')
    fea = data['fea']
    fea = fea.astype(np.float64)
    labels = data['gnd'] - 1
    labels = labels.flatten()
    labels[labels == 255] = 9
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

if __name__ == '__main__':
    data = np.loadtxt('E:/select1m.csv', delimiter=',')
    t = data[:, 0].astype(np.int)
    d = data[:, 1:]
    gc.collect()
    lib = np.loadtxt('library/HEPMASS_5-10_0.7_0.7_50_FSRSNC_pure.res', delimiter=',')
    adjc = build_hypergraph_adjacency(lib)
    fea = adjc.transpose().tocsr()
    print 'Feature loading completed.'
    l = esc.efficient_spectral_clustering(fea, 2, m=100)
    print Metrics.normalized_max_mutual_info_score(t, l)
    l = esc.efficient_spectral_clustering(fea, 2, m=500)
    print Metrics.normalized_max_mutual_info_score(t, l)
    l = esc.efficient_spectral_clustering(fea, 2, m=1000)
    print Metrics.normalized_max_mutual_info_score(t, l)
    l = esc.efficient_spectral_clustering(fea, 2, m=1500)
    print Metrics.normalized_max_mutual_info_score(t, l)
    l = esc.efficient_spectral_clustering(fea, 2, m=2000)
    print Metrics.normalized_max_mutual_info_score(t, l)


    # d, t = load_skin()
    # print 'Data loading completed.'
    # lib = np.loadtxt('library/SKIN_10-20_0.5_1.0_160_FSRSNC_pure.res', delimiter=',')
    # adjc = build_hypergraph_adjacency(lib)
    # fea = adjc.transpose().tocsr()
    # print 'Feature loading completed.'
    # l = esc.efficient_spectral_clustering(fea, 2)
    # print Metrics.normalized_max_mutual_info_score(t, l)
    # print 'hello'

    # d, t = load_digit()
    # print 'Data loading completed.'
    # lib = np.loadtxt('library/OPTDIGITS_10-100_0.5_0.3_160_FSRSNC_pure.res', delimiter=',')
    # adjc = build_hypergraph_adjacency(lib)
    # fea = adjc.transpose().tocsr()
    # print 'Feature loading completed.'
    # l = esc.efficient_spectral_clustering(fea, 10, m=1000)
    # print Metrics.normalized_max_mutual_info_score(t, l)
    # print 'hello'

    # d, t = load_usps()
    # print 'Data loading completed.'
    # lib = np.loadtxt('library/USPS_50-100_0.5_0.3_160_FSRSNC_pure.res', delimiter=',')
    # adjc = build_hypergraph_adjacency(lib)
    # fea = adjc.transpose().tocsr()
    # print 'Feature loading completed.'
    # l = esc.efficient_spectral_clustering(fea, 10)
    # print Metrics.normalized_max_mutual_info_score(t, l)
    # print 'hello'

    # d, t = load_mnist_4000()
    # print 'Data loading completed.'
    # lib = np.loadtxt('library/MNIST4000_10-100_0.5_0.8_160_FSRSNC.res', delimiter=',')
    # adjc = build_hypergraph_adjacency(lib)
    # fea = adjc.transpose().tocsr()
    # print 'Feature loading completed.'
    # l = esc.efficient_spectral_clustering(fea, 10)
    # print Metrics.normalized_max_mutual_info_score(t, l)
    # print 'hello'


    # d, t = load_covtype()
    # print 'Data loading completed.'
    # lib = np.loadtxt('library/covtype_labels.txt', delimiter=',')
    # adjc = build_hypergraph_adjacency(lib)
    # fea = adjc.transpose().tocsr()
    # print 'Feature loading completed.'
    # l = esc.efficient_spectral_clustering(fea, 7, m=200)
    # print Metrics.normalized_max_mutual_info_score(t, l)
    # l = esc.efficient_spectral_clustering(fea, 7, m=400)
    # print Metrics.normalized_max_mutual_info_score(t, l)
    # l = esc.efficient_spectral_clustering(fea, 7, m=800)
    # print Metrics.normalized_max_mutual_info_score(t, l)
    # l = esc.efficient_spectral_clustering(fea, 7, m=1600)
    # print Metrics.normalized_max_mutual_info_score(t, l)
    # l = esc.efficient_spectral_clustering(fea, 7, m=2000)
    # print Metrics.normalized_max_mutual_info_score(t, l)
    # print 'hello'

    # d, t = load_mnist_full()
    # print 'Data loading completed.'
    # lib = np.loadtxt('library/MNIST_FULL_10-100_0.7_0.5_160_FSRSNC_pure.res', delimiter=',')
    # adjc = build_hypergraph_adjacency(lib)
    # fea = adjc.transpose().tocsr()
    # print 'Feature loading completed.'
    # l = esc.efficient_spectral_clustering(d, 10)
    # print Metrics.normalized_max_mutual_info_score(t, l)
    # print 'hello'
