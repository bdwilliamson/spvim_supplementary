import numpy as np

import data_gen_funcs


class Dataset:
    """
    Stores data
    """
    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


class DataGenerator:
    """
    Conditional expectation functions
    """
    def __init__(self, func_name, n_train, n_test, num_p, binary, beta, cor = None):
        self.n_train = n_train
        self.n_test = n_test
        self.num_p = num_p
        self.func = getattr(data_gen_funcs, func_name)
        self.beta = beta
        self.binary = binary
        self.cor = cor

    def create_data(self):
        x_train, y_train = self._create_data(self.n_train)
        x_test, y_test = self._create_data(self.n_test)
        return Dataset(x_train, y_train, x_test, y_test)

    def _create_data(self, size_n):
        if size_n <= 0:
            return None, None, None
        mu = np.zeros((self.num_p,))
        Sigma = np.zeros((self.num_p, self.num_p))
        np.fill_diagonal(Sigma, 1)
        if self.num_p > 10:
            # X1, X11 correlated
            Sigma[1 - 1, 11 - 1] = self.cor[0]
            Sigma[11 - 1, 1 - 1] = self.cor[0]
            Sigma[3 - 1, 12 - 1] = self.cor[1]
            Sigma[12 - 1, 3 - 1] = self.cor[1]
            Sigma[3 - 1, 13 - 1] = self.cor[1]
            Sigma[13 - 1, 3 - 1] = self.cor[1]
            Sigma[5 - 1, 14 - 1] = self.cor[2]
            Sigma[14 - 1, 5 - 1] = self.cor[2]

        xs = np.random.multivariate_normal(mean = mu, cov = Sigma, size = size_n)
        ## make the outcome
        if self.binary:
            true_mean = self.func(xs, self.beta)
            true_mean = np.reshape(true_mean, (true_mean.size, 1))
            y = np.random.binomial(n = 1, p = true_mean)
        else:
            true_mean = self.func(xs, self.beta)
            true_mean = np.reshape(true_mean, (true_mean.size, 1))
            y = np.random.normal(loc = true_mean, scale = 1)

        return xs, y
