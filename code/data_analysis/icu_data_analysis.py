#!/usr/local/bin/python3
## analyze the ICU data

## --------------------------------------------------
## load libraries and user-defined functions
## --------------------------------------------------
import numpy as np
import argparse
import time
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import shap
from scipy.stats import norm
import pandas as pd
from warnings import warn
import vimpy
import lime

import utils as uts
import data_generator as dg
import measures_of_predictiveness as mp
import get_influence_functions as gif
import compute_ic as ci
import get_shapley_value as gsv
import shapley_hyp_test as sht

## set up args
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, help = "Path to dataset", default = '../icu_data/icu_data_processed.pkl')
parser.add_argument("--seed", type = int, help = "Random number seed", default = 4747)
parser.add_argument("--measure", type = str, help = "Measure of predictiveness to use", default = 'auc')
parser.add_argument("--output-dir", type = str, help = "Where to save off shapley values", default = '../results/')
parser.add_argument("--estimator-type", type = str, help = "estimator to fit", default = "nn")
args = parser.parse_args()
print("Running " + args.estimator_type + " for VIM measure " + args.measure)
## --------------------------------------------------
## load the data, set up
## --------------------------------------------------
data = uts.pickle_from_file(args.dataset)
p = data.x_train.shape[1]
np.random.seed(args.seed)
folds_outer = np.random.choice(a = np.arange(2), size = data.y_train.shape[0], replace = True, p = np.array([0.25, 0.75]))
data_0 = dg.Dataset(x_train = data.x_train[folds_outer == 0, :], y_train = data.y_train[folds_outer == 0], x_test = None, y_test = None)
data_1 = dg.Dataset(x_train = data.x_train[folds_outer == 1, :], y_train = data.y_train[folds_outer == 1], x_test = None, y_test = None)
cc_all = (np.sum(np.isnan(data.x_train), axis = 1) == 0)
cc_all_test = (np.sum(np.isnan(data.x_test), axis = 1) == 0)

if args.measure == 'auc':
    measure_func = mp.auc
    objective_function = 'binary:logistic'
    sl_scorer = log_loss
    mlp_class = MLPClassifier
    pred_type = "classification"
    ensemble_method = StackingClassifier
    stack_combiner = LogisticRegression
    stack_method = 'predict_proba'
else:
    measure_func = mp.r_squared
    objective_function = 'reg:linear'
    sl_scorer = mean_squared_error
    mlp_class = MLPRegressor
    pred_type = "regression"
    ensemble_method = StackingRegressor
    stack_combiner = LinearRegression
    stack_method = 'predict'

## set up the cross-validated selector
if "tree" in args.estimator_type:
    print("Fitting boosted trees")
    ntrees = np.arange(2000, 14000, 2000)
    param_grid = [{'n_estimators': ntrees}]
    ## create CV objects
    cv_est = GridSearchCV(XGBRegressor(objective = objective_function, max_depth = 4, verbosity = 0, reg_lambda = 0, learning_rate = 0.001), param_grid = param_grid, cv = 5)
    cv_est.fit(data.x_train, data.y_train)
    ensemble = XGBRegressor(objective = objective_function, max_depth = 4, reg_lambda = 0, learning_rate = 0.001, verbosity = 0, n_estimators = cv_est.best_params_['n_estimators'])
    print("Num. estimators:" + str(cv_est.best_params_['n_estimators']))
elif "nn" in args.estimator_type:
    print("Fitting a neural network")
    hidden_layer_sizes = [(37,25,25,20,10,1), (37,25,20,1), (37,25,20,20,1)]
    param_grid = {'hidden_layer_sizes': hidden_layer_sizes}
    num_est = 5
    seeds = np.arange(num_est)
    np.random.seed(5678)
    cv_est = GridSearchCV(mlp_class(activation = 'relu', solver = 'adam', max_iter = 2000, alpha = 0.1), param_grid = param_grid, cv = 5)
    cv_est.fit(data.x_train[cc_all, :], np.ravel(data.y_train[cc_all]))
    est_lst_cv = [('nn_' + str(x), mlp_class(activation = 'relu', solver = 'adam', max_iter = 2000, alpha = 1e-1, hidden_layer_sizes = cv_est.best_params_['hidden_layer_sizes'], shuffle = True, random_state = x)) for x in range(num_est)]
    ensemble = ensemble_method(est_lst_cv, cv = 2, stack_method = stack_method)
    print("Architecture: " + str(cv_est.best_params_['hidden_layer_sizes']))

m = int(data.x_train.shape[0])
gamma = 1
cols = ['feature', 'est', 'se', 'cil', 'ciu', 'p_value', 'hyp_test', 'measure', 'num_subsets_sampled']
output_df, output_df_mean_abs_shap = pd.DataFrame(index = range(int(p)), columns = cols), pd.DataFrame(index = range(int(p)), columns = cols)
output_df_lime, output_df_vim = pd.DataFrame(index = range(int(p)), columns = cols), pd.DataFrame(index = range(int(p)), columns = cols)

## ----------------------------------------------
## (1) get estimates of sampled conditional means
## ----------------------------------------------
print("Obtaining estimates of predictiveness")
## get a list of n subset sizes, Ss, Zs
max_subset = np.array(list(range(p)))
sampling_weights = np.append(np.append(1, [uts.choose(p - 2, s - 1) ** (-1) for s in range(1, p)]), 1)
subset_sizes = np.random.choice(np.arange(0, p + 1), p = sampling_weights / sum(sampling_weights), size = data_1.x_train.shape[0], replace = True)
S_lst_all = [np.sort(np.random.choice(np.arange(0, p), subset_size, replace = False)) for subset_size in list(subset_sizes)]
## only need to continue with the unique subsets S
Z_lst_all = [np.in1d(max_subset, S).astype(np.float64) for S in S_lst_all]
Z, z_counts = np.unique(np.array(Z_lst_all), axis = 0, return_counts = True)
Z_lst = list(Z)
Z_aug_lst = [np.append(1, z) for z in Z_lst]
S_lst = [max_subset[z.astype(bool).tolist()] for z in Z_lst]
## get v, preds, ic for each unique S
preds_none = np.repeat(np.mean(data_1.y_train), data_1.x_train.shape[0])
v_none = measure_func(data_1.y_train, preds_none)
ic_none = ci.compute_ic(data_1.y_train, preds_none, args.measure)
## get best_estimators and set up either an ensemble or a prediction function for each s
if 'nn' in args.estimator_type:
    ccs = [(np.sum(np.isnan(data_1.x_train[:, s]), axis = 1) == 0) for s in S_lst[1:]]
    best_param_lst = [cv_est.fit(data_1.x_train[:, S_lst[1:][i]][ccs[i], :], np.ravel(data_1.y_train[ccs[i]])).best_params_['hidden_layer_sizes'] for i in range(len(S_lst[1:]))]
    ensemble_funcs = [ensemble_method([('nn_' + str(x), mlp_class(activation = 'relu', solver = 'adam', max_iter = 2000, alpha = 1e-1, hidden_layer_sizes = params, shuffle = True, random_state = x)) for x in range(num_est)], cv = 2, stack_method = stack_method) for params in best_param_lst]
else:
    ensemble_funcs = [ensemble for i in range(len(S_lst[1:]))]
## get v, preds, ic for the remaining non-null groups
start = time.time()
v_lst, preds_lst, ic_lst, folds_lst = zip(*(mp.cv_predictiveness(data_1, S_lst[1:][i], measure_func, ensemble_funcs[i], V = 5, stratified = True, na_rm = True, type = pred_type) for i in range(len(S_lst[1:]))))
end = time.time()
print("Estimating predictiveness took " + str(end - start) + " seconds")
v_lst_all = [v_none] + list(v_lst)
preds_lst_all = [preds_none] + list(preds_lst)
ic_lst_all = [ic_none] + list(ic_lst)
uts.pickle_to_file(v_lst_all, args.output_dir + 'vs_' + args.measure + '_est_' + args.estimator_type + '.pkl')
uts.pickle_to_file(preds_lst_all, args.output_dir + 'preds_' + args.measure + '_est_' + args.estimator_type + '.pkl')
uts.pickle_to_file(ic_lst_all, args.output_dir + 'ics_' + args.measure + '_est_' + args.estimator_type + '.pkl')
## set up Z, v, W, G, c_n matrices
Z = np.array(Z_aug_lst)
v = np.array(v_lst_all)
W = np.diag(z_counts / np.sum(z_counts))
G = np.vstack((np.append(1, np.zeros(p)), np.ones(p + 1)))
c_n = np.array([v_none, v_lst_all[len(v_lst)]])

## --------------------------------------------------
## estimate the shapley value
## --------------------------------------------------
print("Estimating Shapley values")
## do constrained least squares
A_W = np.sqrt(W).dot(Z)
v_W = np.sqrt(W).dot(v)
kkt_matrix_11 = 2 * A_W.transpose().dot(A_W)
kkt_matrix_12 = G.transpose()
kkt_matrix_21 = G
kkt_matrix_22 = np.zeros((kkt_matrix_21.shape[0], kkt_matrix_12.shape[1]))
kkt_matrix = np.vstack((np.hstack((kkt_matrix_11, kkt_matrix_12)), np.hstack((kkt_matrix_21, kkt_matrix_22))))
ls_matrix = np.vstack((2 * A_W.transpose().dot(v_W.reshape((len(v_W), 1))), c_n.reshape((c_n.shape[0], 1))))
ls_solution = np.linalg.inv(kkt_matrix).dot(ls_matrix)
shapley_vals = ls_solution[0:(p + 1), :]
lambdas = ls_solution[(p + 1):ls_solution.shape[0], :]
uts.pickle_to_file(shapley_vals, args.output_dir + 'shapley_vals_' + args.measure + '_est_' + args.estimator_type + '.pkl')

## ----------------------------------------
## (3) hypothesis Testing
## ----------------------------------------
print("Running hypothesis tests")
## get relevant objects
shapley_ics = gif.shapley_influence_function(Z, z_counts, W, v, shapley_vals, G, c_n, np.array(ic_lst_all), measure_func.__name__)
## if any shapley values are < 0, make zero and print a warning
if any(shapley_vals < 0):
    if any(shapley_vals[1:]):
        warn("At least one estimated shapley value is < 0. Setting to 0.")
    shapley_vals[shapley_vals < 0] = 0

preds_none_0 = np.repeat(np.mean(data_0.y_train), data_0.x_train.shape[0])
v_none_0 = measure_func(data_0.y_train, preds_none_0)
ic_none_0 = ci.compute_ic(data_0.y_train, preds_none_0, measure_func.__name__)
sigma_none_0 = np.sqrt(np.mean((ic_none_0) ** 2) / data_0.y_train.shape[0])
## get the shapley values + null predictiveness on the first split
sigma_none = np.sqrt(np.mean((ic_none) ** 2) / data_1.y_train.shape[0])
shapley_vals_plus = shapley_vals + shapley_vals[0]
sigmas_one = [np.sqrt(gsv.shapley_se(shapley_ics, i, gamma) ** 2 + sigma_none ** 2) for i in range(1, p + 1)]
test_statistics, p_values, hyp_tests = sht.shapley_hyp_test(shapley_vals_plus[1:], v_none_0, sigmas_one, sigma_none_0, level = 0.05, p = p)

uts.pickle_to_file(test_statistics, args.output_dir + 'test_statistics_measure_' + args.measure + '.pkl')
uts.pickle_to_file(sigmas_one, args.output_dir + 'single_feature_sigmas_measure_' + args.measure + '.pkl')
uts.pickle_to_file(p_values, args.output_dir + 'p_values_measure_' + args.measure + '.pkl')

## save off spvim values
for i in range(1, shapley_vals.shape[0]):
    ## get shapley values, se, ci
    shapley_val, shapley_se, shapley_ci = gsv.get_shapley_value(shapley_vals, shapley_ics, i, level = 0.95, gamma = gamma)
    ## add to dfs
    output_df.iloc[i - 1] = np.array([i, shapley_val[0], shapley_se, shapley_ci[0], shapley_ci[1], p_values.flatten()[i - 1], hyp_tests.flatten()[i - 1], args.measure, Z.shape[0]]).reshape(1, len(cols))

output_df.to_csv(args.output_dir + 'icu_data_analysis_measure_' + str(args.measure) + '_est_' + str(args.estimator_type) + '.csv')
## -------------------------------------
## SHAP values
## -------------------------------------
print("Getting SHAP values")
## fit the full regression function
cc_all = (np.sum(np.isnan(data.x_train), axis = 1) == 0)
cc_all_test = (np.sum(np.isnan(data.x_test), axis = 1) == 0)
start = time.time()
ensemble.fit(data.x_train[cc_all, :], np.ravel(data.y_train[cc_all]))
## print test-set error
if args.measure == "auc":
    if 'nn' in args.estimator_type:
        test_preds = np.mean(ensemble.transform(data.x_test[cc_all_test, :]), axis = 1)
    else:
        test_preds = ensemble.predict_proba(data.x_test[cc_all_test, :])
else:
    test_preds = ensemble.predict(data.x_test[cc_all_test, :])

log_lik = (-1) * sl_scorer(y_true = np.ravel(data.y_test[cc_all_test]), y_pred = test_preds, normalize = False)
print('Estimated negative log likelihood: ' + str(log_lik))
if "tree" in args.estimator_type:
    explainer = shap.TreeExplainer(ensemble)
    shap_values = explainer.shap_values(data.x_test[cc_all_test, :])
else:
    if args.measure == "auc":
        explainer = shap.KernelExplainer(ensemble.transform, shap.kmeans(data.x_train[cc_all, :], 100))
        tmp = explainer.shap_values(data.x_test[cc_all_test, :], nsample = 500)
        shap_values = np.mean(np.array(tmp), axis = 0)
    else:
        explainer = shap.KernelExplainer(ensemble.transform, shap.kmeans(data.x_train[cc_all, :], 100))
        tmp = explainer.shap_values(data.x_test[cc_all_test, :], nsample = 500)
        shap_values = np.mean(np.array(tmp), axis = 0)

end = time.time()
print("Computing SHAP values took " + str(end - start) + " seconds")
uts.pickle_to_file(shap_values, args.output_dir + 'shap_values_measure_' + args.measure + '_est_' + args.estimator_type + '.pkl')

for i in range(1, shapley_vals.shape[0]):
    ## get mean absolute SHAP value
    mean_abs_shap = np.mean(np.absolute(np.sum(shap_values[:, i - 1, None], axis = 1)), axis = 0)
    ## add to dfs
    output_df_mean_abs_shap.iloc[i - 1] = np.array([i, mean_abs_shap, np.nan, np.nan, np.nan, np.nan, np.nan, 'mean_abs_shap', np.nan]).reshape(1, len(cols))

## -------------------------------------
## LIME
## -------------------------------------
print("Getting LIME values")
start = time.time()
np.random.seed(123456)
ensemble.fit(data.x_train[cc_all, :], np.ravel(data.y_train[cc_all]))
lime_x_train, lime_x_test, lime_y_train, lime_y_test = data.x_train[cc_all, :], data.x_test[cc_all_test, :], np.ravel(data.y_train[cc_all]), np.ravel(data.y_test[cc_all_test])
if args.measure == "auc":
    explainer = lime.lime_tabular.LimeTabularExplainer(lime_x_train, feature_names=[str(x) for x in range(p)], class_names='status', discretize_continuous=True, feature_selection='auto')
    if 'nn' in args.estimator_type:
        pred_func = uts.ensemble_pred_func(ensemble)
    else:
        pred_func = ensemble.predict_proba
else:
    pred_func = ensemble.predict
    explainer = lime.lime_tabular.LimeTabularExplainer(lime_x_train, feature_names=[str(x) for x in range(p)], class_names='status', discretize_continuous=True, mode = 'regression', feature_selection='auto')


features_in_lime = np.zeros((data.x_test[cc_all, :].shape))
for i in range(data.x_test[cc_all, :].shape[0]):
    exp_i = explainer.explain_instance(lime_x_test[i, :], pred_func)
    strings_i = [exp[0] for exp in exp_i.as_list()]
    features_i = [uts.get_lime_features(x) for x in strings_i]
    features_in_lime[i, :] = np.array([str(x) in features_i for x in range(p)])

lime_importances = np.mean(features_in_lime, axis = 0)
end = time.time()
print("Computing LIME values took " + str(end - start) + " seconds")
for j in range(1, lime_importances.shape[0] + 1):
    output_df_lime.iloc[j - 1] = np.array([j, lime_importances[j - 1], np.nan, np.nan, np.nan, np.nan, np.nan, 'mean_lime_select', np.nan]).reshape(1, len(cols))

output_df_lime.to_csv(args.output_dir + 'icu_lime_ests_measure_' + str(args.measure) + '_est_' + str(args.estimator_type) + '.csv')
## -------------------------------------
## VIM (conditional)
## -------------------------------------
print("Getting VIM values")
np.random.seed(56789)
cc_data = dg.Dataset(x_train = data.x_train[cc_all, :], y_train = data.y_train[cc_all])
folds_outer_vim = np.random.choice(a = np.arange(2), size = cc_data.y_train.shape[0], replace = True, p = np.array([0.5, 0.5]))
cc_data_vim_0 = dg.Dataset(x_train = cc_data.x_train[folds_outer_vim == 0, :], y_train = cc_data.y_train[folds_outer_vim == 0], x_test = None, y_test = None)
cc_data_vim_1 = dg.Dataset(x_train = cc_data.x_train[folds_outer_vim == 1, :], y_train = cc_data.y_train[folds_outer_vim == 1], x_test = None, y_test = None)
start = time.time()
v_full, preds_full, ic_full, folds_full, cc_full = vimpy.cv_predictiveness(cc_data_vim_1.x_train, cc_data_vim_1.y_train, S = np.arange(p), measure = measure_func, pred_func = ensemble, V = 5, stratified = False)
for s in range(p):
    indices = np.delete(np.arange(p), s)
    if "tree" in args.estimator_type:
        cv_small = GridSearchCV(XGBRegressor(objective = objective_function, max_depth = 4, verbosity = 0, reg_lambda = 0, learning_rate = 0.001), param_grid = param_grid, cv = 5)
        cv_small.fit(cc_data_vim_0.x_train[:, indices], np.ravel(cc_data_vim_0.y_train))
        ensemble_s = XGBRegressor(objective = objective_function, max_depth = 4, verbosity = 0, reg_lambda = 0, learning_rate = 0.001, n_estimators = cv_small.best_params_['n_estimators'])
    else:
        cv_small = GridSearchCV(mlp_class(activation = 'relu', solver = 'adam', max_iter = 2000, alpha = 0.1), param_grid = param_grid, cv = 5)
        cv_small.fit(cc_data_vim_0.x_train[:, indices], np.ravel(cc_data_vim_0.y_train))
        est_lst_cv_s = [('nn_' + str(x), mlp_class(activation = 'relu', solver = 'adam', max_iter = 2000, alpha = 1e-1, hidden_layer_sizes = cv_small.best_params_['hidden_layer_sizes'], shuffle = True, random_state = x)) for x in range(num_est)]
        if args.measure == "auc":
            ensemble_s = ensemble_method(est_lst_cv_s, cv = 2, stack_method = stack_method)
        else:
            ensemble_s = ensemble_method(est_lst_cv_s, cv = 2)
    v_redu, preds_redu, ic_redu, folds_redu, cc_redu = vimpy.cv_predictiveness(cc_data_vim_0.x_train, cc_data_vim_0.y_train, S = indices, measure = measure_func, pred_func = ensemble_s, V = 5, stratified = False)
    these_folds = [folds_outer_vim, folds_full, folds_redu]
    vimp_cv = vimpy.cv_vim(y = cc_data.y_train, x = cc_data.x_train, s = s, f = preds_full, r = preds_redu, V = 5, measure_type = args.measure, na_rm = True, folds = these_folds)
    ## get the point estimate
    vimp_cv.get_point_est()
    ## get the standard error
    vimp_cv.get_influence_function()
    vimp_cv.get_se()
    ## get a confidence interval
    vimp_cv.get_ci()
    ## do a hypothesis test, compute p-value
    vimp_cv.hypothesis_test(alpha = 0.05, delta = 0)
    ## display estimates, etc.
    output_df_vim.iloc[s] = np.array([s + 1, vimp_cv.vimp_, vimp_cv.se_, np.ravel(vimp_cv.ci_)[0], np.ravel(vimp_cv.ci_)[1], vimp_cv.p_value_, vimp_cv.hyp_test_, args.measure, np.nan]).reshape(1, len(cols))


end = time.time()
print("Computing VIM values took " + str(end - start) + " seconds")
output_df_vim.to_csv(args.output_dir + 'icu_vim_ests_measure_' + str(args.measure) + '_est_' + str(args.estimator_type) + '.csv')
## -------------------------------------
## end, save results
## -------------------------------------

## concatenate dfs together
all_output_df = pd.concat([output_df, output_df_mean_abs_shap, output_df_lime, output_df_vim], ignore_index = True)
## save it off
all_output_df.to_csv(args.output_dir + 'icu_data_analysis_measure_' + str(args.measure) + '_est_' + str(args.estimator_type) + '.csv')
