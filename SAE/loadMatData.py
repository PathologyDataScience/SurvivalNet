__author__ = 'Song'
import scipy.io as sio
import matlab.engine
import numpy as np
import cPickle


VA = ('data/VA.mat')
LUAD_P = 'data/LUAD_P.mat'
LUSC_P = 'data/LUSC_P.mat'
Brain_P = 'data/Brain_P.mat'


def save_pickle(name='LUAD_P.pickle', p=LUAD_P):
    observed, x, y, at_risk = load_data(p=p)
    f = file(name, 'wb')
    for obj in [observed, x, y, at_risk]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def read_pickle(name='LUSC_P.pickle'):
    f = file(name, 'rb')
    loaded_objects = []
    for i in range(4):
        loaded_objects.append(cPickle.load(f))
    observed, X, survival_time, at_risk_X = loaded_objects
    f.close()
    return observed, X, survival_time, at_risk_X


def augment_data(train_X, train_T, train_C):
    Xa = np.zeros((train_X.shape[0]*2, train_X.shape[1]))
    for i in xrange(train_X.shape[1]):
        if len(np.unique(train_X[:, i])) > 3:
            noise = np.random.uniform(-0.2, 0.2, train_X.shape[0])
            noise_X = train_X[:, i] + noise
            Xa[:, i] = np.concatenate((train_X[:, i], noise_X))
        else:
            Xa[:, i] = np.concatenate((train_X[:, i], train_X[:, i]))
    return Xa, train_T * 2, train_C * 2


def load_augment_data(p=Brain_P):
    mat = sio.loadmat(p)
    eng = matlab.engine.start_matlab()
    X = mat['X']
    C = mat['C']
    T = mat['T']
    test_size = len(X) / 3
    train_X = X[test_size:]
    train_T = [t[0] for t in T[test_size:]]
    train_C = [c[0] for c in C[test_size:]]
    test_X = X[:test_size]
    test_T = np.asarray([t[0] for t in T[:test_size]])
    test_C = np.asarray([c[0] for c in C[:test_size]], dtype='int32')
    train_X, train_T, train_C = augment_data(train_X, train_T, train_C)
    survival_time = train_T
    mat_T = matlab.double(survival_time)
    survival_time, order = eng.sort(mat_T, nargout=2)
    order = np.asarray(order[0]).astype(int) - 1
    temp = matlab.double(survival_time[0])
    at_risk_X = np.asarray(eng.ismember(temp, temp, nargout=2)[1][0]).astype(int) - 1
    train_T = np.asarray(survival_time[0])
    train_O = 1 - np.asarray(train_C, dtype='int32')[order]
    return train_X[order], train_T, train_O, at_risk_X, test_X, test_T, 1 - test_C


def load_data(p=LUAD_P):
    mat = sio.loadmat(p)
    X = mat['X']
    C = mat['C']
    T = mat['T']
    eng = matlab.engine.start_matlab()
    survival_time = [t[0] for t in T]
    mat_T = matlab.double(survival_time)
    survival_time, order = eng.sort(mat_T, nargout=2)
    order = np.asarray(order[0]).astype(int) - 1
    temp = matlab.double(survival_time[0][:])
    at_risk = np.asarray(eng.ismember(temp, temp, nargout=2)[1][0]).astype(int)
    censored = np.asarray([c[0] for c in C], dtype='int32')
    survival_time = np.asarray(survival_time[0])
    # print survival_time
    return 1 - censored[order], X[order], survival_time, at_risk - 1

if __name__ == '__main__':
    # save_pickle(name='VA.pickle', p=VA)
    # save_pickle(name='LUSC_P.pickle', p=LUSC_P)
    # save_pickle(name='Brain_P.pickle', p=Brain_P)
    load_augment_data()