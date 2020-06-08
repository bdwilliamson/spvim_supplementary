## hypothesis testing with shapley values


def shapley_hyp_test(vs_one_1, v_none_0, sigmas_one, sigma_none, delta = 0, level = 0.05, p = 3):
    """
    Hypothesis testing for Shapley values

    @param vs_one_1: one-feature measures of predictiveness
    @param v_none_0: null-model predictiveness
    @param sigmas_one: ses
    @param sigma_none: null-model se
    @param delta: value for testing
    @param level: significance level

    @return: test_statistics (the test statistics), p_vals (p-values), hyp_tests (the hypothesis testing results)
    """
    import numpy as np
    from scipy.stats import norm

    test_statistics = [(vs_one_1[v] - v_none_0 - delta) / (np.sqrt(sigmas_one[v] ** 2 + sigma_none ** 2)) for v in range(p)]
    p_values = 1. - norm.cdf(test_statistics)
    hyp_tests = p_values < level
    return test_statistics, p_values, hyp_tests
