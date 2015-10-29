import scipy.io as sio
import matlab.engine
import numpy as np

VA = 'data/VA.mat'
LUAD_P = 'data/LUAD_P.mat'
LUSC_P = 'data/LUSC_P.mat'
Brain_P = 'data/Brain_P.mat'


def discrete_time_data(old_x, observed, survival_time, start=0.1):
    x = []
    new_observed = []
    # each entry in x is a list of all time prior than x
    for index in xrange(len(old_x)):
        data = old_x[index]
        temp = list(data)
        time = survival_time[index]
        step = start
        while step < time:
            new_row = temp[:]
            new_row[0] = step
            x.append(new_row)
            new_observed.append(0.0)
            step += start
        temp[0] = step
        x.append(temp)
        if observed[index]:
            new_observed.append(1.0)
        else:
            new_observed.append(0.0)
    return np.asarray(x), np.asarray(new_observed)


def load_data(p=Brain_P, step=0.1):
    print "loading data..."
    observed, X, survival_time = load_mat_data(p)
    test_size = len(X) / 3
    train_X = X[test_size:]
    # print train_X.shape
    train_y = survival_time[test_size:]
    train_X, train_observed = discrete_time_data(train_X, observed, train_y, start=step)
    test_X = X[:test_size]
    test_observed = observed[:test_size]
    test_y = survival_time[:test_size]
    print train_X.shape
    return train_X, train_observed, test_y, test_observed, test_X


def load_mat_data(p):
    mat = sio.loadmat(p)
    X = mat['X']
    C = mat['C']
    T = mat['T']
    eng = matlab.engine.start_matlab()
    survival_time = [t[0] for t in T]
    mat_T = matlab.double(survival_time)
    survival_time, order = eng.sort(mat_T, nargout=2)
    order = np.asarray(order[0]).astype(int) - 1
    censored = np.asarray([c[0] for c in C], dtype='int32')
    survival_time = np.asarray(survival_time[0])
    # print survival_time
    return 1 - censored[order], X[order].astype(float), survival_time


def save_csv(name="LUAD_P.csv", p=LUAD_P):
    observed, X, survival_time = load_mat_data(p=p)
    print X.shape
    X = np.insert(X, 0, observed, axis=1)
    X = np.insert(X, 0, survival_time, axis=1)
    print X.shape
    np.savetxt(Path('C:/Users/Song/Research/biomed/Survival/RSF/' + name), X, delimiter=',', fmt='%10.5f')

if __name__ == '__main__':
    save_csv(name="LUSC_P.csv", p=LUSC_P)
    save_csv(name="VA.csv", p=VA)
    save_csv(name="Brain_P.csv", p=Brain_P)
