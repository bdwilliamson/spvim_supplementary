import numpy as np


def expit(x):
    return np.exp(x) / (1. + np.exp(x))


def logistic_gam(x, beta):
    return beta[0] * x[:, 0] + beta[1] * x[:, 1] ** 2 + beta[5] * x[:, 5] ** 3


def step_function(x):
    return 0 + (-1) * 6 * (x <= -4) + (-1) * 4 * (x > -4) * (x <= -2) + (-1) * 2 * (x > -2) * (x <= 0) + 2 * (x > 2) * (x <= 4) + 4 * (x > 4)


def wiggle(x):
    return 0 + (-1) * (x <= -4) + (-1) * (x > -2) * (x <= 0) + (-1) * (x > 2) * (x <= 4) + 1 * (x > -4) * (x <= -2) + 1 * (x > 0) * (x <= 2) + 1 * (x > 4)


def ten_variable_binary_conditional_mean(x, beta):
    return expit(logistic_gam(x, beta))


def ten_variable_continuous_conditional_mean(x, beta):
    return np.sign(x[:, 0]) + step_function(x[:, 2]) + wiggle(x[:, 4])


def lm_conditional_mean(x, beta):
    return np.dot(x, beta)
