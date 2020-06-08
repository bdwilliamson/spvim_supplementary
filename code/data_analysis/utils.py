## utility functions
import pickle
import pandas
import math


def pickle_to_file(obj, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f, protocol=-1)


def pickle_from_file(file_name):
    with open(file_name, "rb") as f:
        out = pickle.load(f)
    return out


def read_pickle_file(file_name):
    pickle_data = pandas.read_pickle(file_name)
    return pickle_data


def get_binary_representation(num, p):
    """
    @param num: the number
    @param p: the number of digits to go out to
    """
    return '{num:0{p}b}'.format(num = num, p = p)[::-1]


def find_all(s, c):
    idx = s.find(c)
    while idx != -1:
        yield idx
        idx = s.find(c, idx + 1)


def choose(n, k):
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n - k)))


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    from itertools import combinations, chain
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def create_z_row(power_set_i, p):
    """
    Create a row of Z for a given subset of the power set of p

    @param power_set_i: the ith row of the power set
    @param p: the number of covariates

    @return: a vector of 0's and 1's (1 corresponds to index in s)
    """
    return [int(x in power_set_i) for x in range(p)]


def make_w(p, s):
    """
    Create w(s)

    @param p: the total number of covariates
    @param s: the size of the subset

    @return w(s)
    """
    ## get the sampling weight
    if (s == 0) or (s == p):
        sampling_weight = 1
    else:
        sampling_weight = choose(p - 2, s - 1) ** (-1)
    return sampling_weight


def create_z_w(Z):
    """
    Create Z_m matrix (m by (p+1)) and W_m matrix (m by m)
    based on a list of Z values created by SGD

    @param Z: a numpy array of Z values

    @return: a dictionary with Z_m and W_m
    """
    import numpy as np

    m = Z.shape[0]
    p = Z.shape[1]
    num_ones = np.sum(Z, axis = 1)
    Z_m = np.hstack((np.repeat(1, m).reshape(m, 1), Z))
    W_m = np.zeros((m, m))
    W_m[np.diag_indices(m)] = [make_w(p, s) for s in num_ones]
    # sorted_z = np.take(Z, num_ones.argsort(), axis = 0)
    return {'Z': Z_m, 'W': W_m}


def make_folds(draw, V, stratified = True):
    """
    Create folds for CV (potentially stratified)
    """
    import numpy as np
    if stratified:
        y_1 = draw.y_train == 1
        y_0 = draw.y_train == 0
        folds_1 = np.resize(np.arange(V), sum(y_1))
        np.random.shuffle(folds_1)
        folds_0 = np.resize(np.arange(V), sum(y_0))
        np.random.shuffle(folds_0)
        folds = np.empty((draw.y_train.shape[0]))
        folds[np.ravel(y_1)] = folds_1
        folds[np.ravel(y_0)] = folds_0
    else:
        folds = np.resize(np.arange(V), draw.y_train.shape[0])
        np.random.shuffle(folds)
    return folds


def create_kkt_matrix(A_W, G):
    """
    Create KKT matrix
    @param A_W the main matrix
    @param G the constraint matrix
    """
    import numpy as np
    kkt_matrix_11 = 2 * A_W.transpose().dot(A_W)
    kkt_matrix_12 = G.transpose()
    kkt_matrix_21 = G
    kkt_matrix_22 = np.zeros((kkt_matrix_21.shape[0], kkt_matrix_12.shape[1]))
    kkt_matrix = np.vstack((np.hstack((kkt_matrix_11, kkt_matrix_12)), np.hstack((kkt_matrix_21, kkt_matrix_22))))
    return kkt_matrix


def ensemble_pred_func(ensemble):

    def mean_transform(x):
        import numpy as np
        probs = np.mean(ensemble.transform(x), axis = 1)
        return np.vstack((1 - probs, probs)).T
    return mean_transform


def tree_binary_pred_func(ensemble):

    def multiclass_preds(x):
        import numpy as np
        probs = ensemble.predict(x)
        return np.vstack((1 - probs, probs)).T
    return multiclass_preds


def get_lime_features(string):

    import re
    split_string = re.sub("=", "", re.sub("<", "", string)).split(" ")
    if len(split_string) > 3:
        ret = split_string[2]
    else:
        ret = split_string[0]
    return ret
