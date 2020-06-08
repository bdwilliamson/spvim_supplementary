## get shapley value, conditional importance


def get_shapley_value(shapley_values, shapley_ics, idx, level = 0.95, gamma = 1, na_rm = True):
    """
    Return the idxth Shapley value and its CI from a list of all Shapley values

    @param shapley_values: all Shapley values (may be either all individual covariates or a group relative to individual covariates)
    @param shapley_ics: all Shapley ICs (a dictionary)
    @param idx: the index of interest
    @param level: CI level
    @param gamma: value for sampling
    @param na_rm: remove NaNs?

    @return the Shapley value of interest and its CI
    """

    ## if s is a list of values, sum over them; otherwise, return the sth value
    point_est = shapley_point_est(shapley_values, idx)
    se = shapley_se(shapley_ics, idx, gamma)
    ci = shapley_ci(point_est, se, level)
    return point_est, se, ci[0]


def shapley_point_est(shapley_vals, idx):
    """
    Shapley value point estimate for the current index

    @param shapley_vals: all shapley vals
    @param idx: the index of interest

    @return shapley value point estimate corresponding to idx
    """
    shapley_point_est = shapley_vals[idx]
    return shapley_point_est


def conditional_vim_point_est(mps, full_idx, redu_idx):
    """
    Conditional VIM point estimate based on measures of predictiveness

    @param ms: measures of predictiveness
    @param full_idx, redu_idx: the indices of the correct measures of predictiveness to compare

    @return the point estimate of conditional vim
    """
    vim_point_est = all_mps[full_idx] - all_mps[redu_idx]
    return vim_point_est


def shapley_ci(point_est, se, level):
    """
    CI for a shapley value

    @param point_est: the point estimate
    @param se: the standard error
    @param level: the significance level
    """
    import numpy as np
    from scipy.stats import norm

    ## compute the quantiles for the CI
    a = np.array([(1 - level) / 2, 1 - (1 - level) / 2])
    fac = norm.ppf(a)

    ci = point_est + np.outer((se), fac)
    return ci


def shapley_se(shapley_ics, idx, gamma, na_rm = True):
    """
    Standard error for the desired Shapley value

    @param shapley_ics: all influence function estimates
    @param idx: the index of interest
    @param gamma: the constant for sampling
    @param na_rm: remove NaNs?

    @return the standard error corresponding to the shapley value at idx
    """
    import numpy as np
    if na_rm:
        var_v = np.nanvar(shapley_ics['contrib_v'][idx, :])
        var_s = np.nanvar(shapley_ics['contrib_s'][idx, :])
    else:
        var_v = np.var(shapley_ics['contrib_v'][idx, :])
        var_s = np.var(shapley_ics['contrib_s'][idx, :])
    se = np.sqrt(var_v / shapley_ics['contrib_v'].shape[1] + var_s / shapley_ics['contrib_s'].shape[1] / gamma)
    return se


def shapley_sgd_se(betas, idx):
    """
    Standard error for the desired Shapley value, based on SGD

    @param betas: the influence functions from SGD
    @param idx: the index of interest

    @return the standard error corresponding to the Shapley value at idx, based on SGD
    """
    import numpy as np
    est_var = np.mean(betas[idx, :] ** 2)
    se = np.sqrt(est_var / betas.shape[1])
    return se
