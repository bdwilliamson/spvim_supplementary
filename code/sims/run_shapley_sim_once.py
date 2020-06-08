# run through simulation one time


def do_one(n_train, n_test, p, m, measure_type, binary, gamma, cor, V, conditional_mean = "nonlinear", estimator_type = "tree"):
    """
    Run the simulation one time for a given set of parameters

    @param n: sample size
    @param p: dimension
    @param m: number of subsets to sample for SGD
    @param tail: number of SGD samples to use for tail averaging
    @param measure_type: variable importance measure
    @param binary: is the outcome binary?
    @param gamma: the constant multiplied by n for sampling
    @param cor: the correlation (only used if p > 10)
    @param V: folds for cross-fitting
    @param conditional_mean: type of conditional mean (linear or nonlinear)
    @param estimator_type: the type of estimator to fit (tree or linear model)

    @return multiple values, including
        shapley_vals: the shapley values
        shapley_ics: the influence curves for the shapley values
        shap_values: the mean absolute SHAP values
        shapley_dict['num_subsets_sampled']: the number of subsets sampled
        all_mps: all measures of predictiveness
        p_values: p-values
        hyp_tests: hypothesis test decisions
        shapley_dict['beta']: the "beta" matrix, from SGD on ics
    """
    # import standard libraries
    import numpy as np
    from xgboost import XGBRegressor
    from sklearn.linear_model import LinearRegression
    import shap
    from sklearn.model_selection import GridSearchCV
    from warnings import warn

    # import user-defined functions
    import data_generator as dg
    import measures_of_predictiveness as mp
    import utils as uts
    import get_influence_functions as gif
    import compute_ic as ci
    import get_shapley_value as gsv
    import shapley_hyp_test as sht

    # generate data
    if conditional_mean == "nonlinear":
        if binary:
            func_name = "ten_variable_binary_conditional_mean"
        else:
            func_name = "ten_variable_continuous_conditional_mean"
    else:
        func_name = "lm_conditional_mean"

    beta = np.array([1, 0, 1.2, 0, 1.05, 0] + [0] * (p - 6))

    if measure_type == "r_squared":
        measure_func = mp.r_squared
        objective_function = 'reg:linear'
    else:
        measure_func = mp.auc
        objective_function = 'binary:logistic'

    data_gen = dg.DataGenerator(func_name, n_train, n_test, p, binary, beta, cor)
    draw = data_gen.create_data()
    folds_outer = np.random.choice(a = np.arange(2), size = draw.y_train.shape[0], replace = True, p = np.array([0.25, 0.75]))
    draw_0 = dg.Dataset(x_train = draw.x_train[folds_outer == 0, :], y_train = draw.y_train[folds_outer == 0], x_test = None, y_test = None)
    draw_1 = dg.Dataset(x_train = draw.x_train[folds_outer == 1, :], y_train = draw.y_train[folds_outer == 1], x_test = None, y_test = None)
    # set up args for xgboost

    # use the cross-validated selector to get the number of trees
    ntrees_tree = np.array([50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000])
    lambdas_tree = np.array([1e-3, 1e-2, 1e-1, 1, 5, 10])
    param_grid_tree = [{'n_estimators': ntrees_tree, 'reg_alpha': lambdas_tree}]
    # estimate full regression
    if estimator_type == "tree":
        cv_tree = GridSearchCV(XGBRegressor(objective = objective_function, max_depth = 1, verbosity = 0, learning_rate = 1e-2, reg_lambda = 0), param_grid = param_grid_tree, cv = 5)
        cv_tree.fit(draw.x_train, np.ravel(draw.y_train))
        ensemble_tree = XGBRegressor(objective = objective_function, max_depth = 1, verbosity = 0, reg_lambda = 0, learning_rate = 1e-2, n_estimators = cv_tree.best_params_['n_estimators'], reg_alpha = cv_tree.best_params_['reg_alpha'])
        ensemble = ensemble_tree
        print("Num. est. in boosted tree: " + str(cv_tree.best_params_['n_estimators']))
    else:
        ensemble = LinearRegression(fit_intercept = False)
    # get a list of n subset sizes, Ss, Zs
    max_subset = np.array(list(range(p)))
    sampling_weights = np.append(np.append(1, [uts.choose(p - 2, s - 1) ** (-1) for s in range(1, p)]), 1)
    subset_sizes = np.random.choice(np.arange(0, p + 1), p = sampling_weights / sum(sampling_weights), size = draw.x_train.shape[0] * gamma, replace = True)
    S_lst_all = [np.sort(np.random.choice(np.arange(0, p), subset_size, replace = False)) for subset_size in list(np.sort(subset_sizes))]
    # only need to continue with the unique subsets S
    Z_lst_all = [np.in1d(max_subset, S).astype(np.float64) for S in S_lst_all]
    Z, z_counts = np.unique(np.array(Z_lst_all), axis = 0, return_counts = True)
    Z_lst = list(Z)
    Z_aug_lst = [np.append(1, z) for z in Z_lst]
    S_lst = [max_subset[z.astype(bool).tolist()] for z in Z_lst]
    if estimator_type == "tree":
        cv_tree_small = GridSearchCV(XGBRegressor(objective = objective_function, max_depth = 1, verbosity = 0, learning_rate = 1e-2, reg_lambda = 0), param_grid = param_grid_tree, cv = 5)
        all_s_sizes = [len(s) for s in S_lst[1:]]
        s_sizes = np.unique(all_s_sizes)
        all_best_tree_lst = [None] * len(S_lst[1:])
        all_best_lambda_lst = [None] * len(S_lst[1:])
        for i in range(s_sizes.shape[0]):
            indx = all_s_sizes.index(s_sizes[i])
            this_s = S_lst[1:][indx]
            cc_i = (np.sum(np.isnan(draw_1.x_train[:, this_s]), axis = 1) == 0)
            these_best_params = cv_tree_small.fit(draw_1.x_train[:, this_s][cc_i, :], np.ravel(draw_1.y_train[cc_i])).best_params_
            all_indices = [index for index, value in enumerate(all_s_sizes) if value == s_sizes[i]]
            all_best_tree_lst = [these_best_params['n_estimators'] if x in all_indices else all_best_tree_lst[x] for x in range(len(all_best_tree_lst))]
            all_best_lambda_lst = [these_best_params['reg_alpha'] if x in all_indices else all_best_lambda_lst[x] for x in range(len(all_best_lambda_lst))]
        ensemble_funcs = [XGBRegressor(objective = objective_function, max_depth = 1, verbosity = 0, reg_lambda = 0, reg_alpha = all_best_lambda_lst[i], learning_rate = 1e-2, n_estimators = all_best_tree_lst[i]) for i in range(len(all_best_tree_lst))]
    else:
        ensemble_funcs = [ensemble for i in range(len(S_lst[1:]))]
    # get v, preds, ic for each unique S
    preds_none = np.repeat(np.mean(draw_1.y_train), draw_1.x_train.shape[0])
    v_none = measure_func(draw_1.y_train, preds_none)
    ic_none = ci.compute_ic(draw_1.y_train, preds_none, measure_type)
    # get v, preds, ic for the remaining non-null groups
    v_lst, preds_lst, ic_lst, folds = zip(*(mp.cv_predictiveness(draw_1, S_lst[1:][i], measure_func, ensemble_funcs[i], V = V, stratified = binary, na_rm = False) for i in range(len(S_lst[1:]))))
    v_lst_all = [v_none] + list(v_lst)
    ic_lst_all = [ic_none] + list(ic_lst)
    # set up Z, v, W, G, c_n matrices
    Z = np.array(Z_aug_lst)
    # constrain v >= 0
    v = np.maximum(np.array(v_lst_all), 0)
    W = np.diag(z_counts / np.sum(z_counts))
    G = np.vstack((np.append(1, np.zeros(p)), np.ones(p + 1) - np.append(1, np.zeros(p))))
    c_n = np.array([v_none, v_lst_all[len(v_lst)] - v_none])
    # do constrained least squares
    A_W = np.sqrt(W).dot(Z)
    v_W = np.sqrt(W).dot(v)
    kkt_matrix = uts.create_kkt_matrix(A_W, G)
    ls_matrix = np.vstack((2 * A_W.transpose().dot(v_W.reshape((len(v_W), 1))), c_n.reshape((c_n.shape[0], 1))))
    ls_solution = np.linalg.pinv(kkt_matrix).dot(ls_matrix)
    shapley_vals = ls_solution[0:(p + 1), :]

    # get relevant objects
    shapley_ics = gif.shapley_influence_function(Z, z_counts, W, v, shapley_vals, G, c_n, np.array(ic_lst_all), measure_func.__name__)
    # if any shapley values are < 0, make zero and print a warning
    if any(shapley_vals < 0):
        if any(shapley_vals[1:] < 0):
            warn("At least one estimated shapley value is < 0. Setting to 0.")
        shapley_vals = np.maximum(shapley_vals, 0)
    if any(shapley_vals > 1):
        if any(shapley_vals[1:] > 1):
            warn("At least one estimated shapley value is > 1. Setting to 1.")
        shapley_vals = np.minimum(shapley_vals, 1)

    # do hypothesis test
    # get the null predictiveness on a separate split
    preds_none_0 = np.repeat(np.mean(draw_0.y_train), draw_0.x_train.shape[0])
    v_none_0 = measure_func(draw_0.y_train, preds_none_0)
    ic_none_0 = ci.compute_ic(draw_0.y_train, preds_none_0, measure_type)
    sigma_none_0 = np.sqrt(np.mean((ic_none_0) ** 2)) / np.sqrt(np.sum(draw_0.y_train.shape[0]))
    # get the shapley values + null predictiveness on the first split
    shapley_vals_plus = shapley_vals + shapley_vals[0]
    sigmas_one = [np.sqrt(gsv.shapley_se(shapley_ics, i, gamma) ** 2 + sigma_none_0 ** 2) for i in range(1, p + 1)]
    test_statistics, p_values, hyp_tests = sht.shapley_hyp_test(shapley_vals_plus[1:], v_none_0, sigmas_one, sigma_none_0, level = 0.05, p = p)

    # get variable importance using SHAP values
    if estimator_type == "tree":
        mod = XGBRegressor(objective = objective_function, learning_rate = 1e-2, reg_lambda = 0, max_depth = 1, n_estimators = cv_tree.best_params_['n_estimators'], reg_alpha = cv_tree.best_params_['reg_alpha'], verbosity = 0)
        mod.fit(draw.x_train, draw.y_train)
        explainer = shap.TreeExplainer(mod)
    else:
        mod = LinearRegression(fit_intercept = False)
        mod.fit(draw.x_train, draw.y_train)
        explainer = shap.LinearExplainer((np.ravel(mod.coef_), 0), draw.x_train, feature_dependence = 'correlation', nsamples = 500)

    shap_values = explainer.shap_values(draw.x_test)

    # return the population shapley values and averaged prediction-level shapley values
    return shapley_vals, shapley_ics, shap_values, Z.shape[0], v, p_values, hyp_tests
