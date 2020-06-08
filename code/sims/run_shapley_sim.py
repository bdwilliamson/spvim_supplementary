#!/usr/local/bin/python3

# simulations for manuscript

# import standard libraries
import numpy as np
import pandas as pd
import time
import os
import argparse

# user-defined functions
import get_shapley_value as gsv
import run_shapley_sim_once as rs

# job id; edit this line if running locally or using a different batch scheduler
job_id = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1

parser = argparse.ArgumentParser()
parser.add_argument("--sim-name", type = str, help = "Name of simulation")
parser.add_argument("--nreps-total", type = int, help = "Number of total replicates")
parser.add_argument("--nreps-per-job", type = int, help = "Number of replicates per job")
parser.add_argument("--V", type = int, help = "number of cross-fitting folds")
parser.add_argument("--est-type", type = str, help = "estimator type (tree or lm)")
parser.add_argument("--conditional-mean", type = str, help = "is conditional mean nonlinear (default) or linear?")
parser.set_defaults(sim_name = "twohundred-var", nreps_total = 1000, nreps_per_job = 5, V = 2, est_type = "tree", conditional_mean = "nonlinear")
args = parser.parse_args()

# set up parameter grid
ns = [500, 1000, 2000, 3000, 4000, 5000]
if "ten" in args.sim_name:
    p = 10
    cor = np.array([0, 0, 0])
elif "fifteen" in args.sim_name:
    p = 15
    cor = np.array([0.7, 0.3, 0.05])
else:
    p = 200
    cor = np.array([0.7, 0.3, 0.05])


# check if binary or not
if "binomial" in args.sim_name:
    binary = True
    measure = 'auc'
else:
    binary = False
    measure = 'r_squared'

# set up unique combinations of s and n
nreps_per_combo = args.nreps_total / args.nreps_per_job
param_lst = [(x, z) for x in ns for z in np.arange(nreps_per_combo)]
# get the current settings
current = {'n': param_lst[job_id][0], 'mc_id': int(param_lst[job_id][1])}
gamma = 2
m = int(gamma * current['n'])
print('Running n = ' + str(current['n']) + ', MC id = ' + str(current['mc_id']))
# set the seed
this_seed = 10 * current['n'] + current['mc_id'] + job_id
print('Seed = ' + str(this_seed))
np.random.seed(this_seed)

# set up output
init_cols = ['mc_id', 'measure', 'n', 's', 'est', 'se', 'cil', 'ciu', 'p_value', 'hyp_test', 'num_subsets_sampled']
mp_cols = np.core.defchararray.add(np.array(['mp_']), np.arange(m).astype('str')).tolist()
cols = init_cols + mp_cols
if p <= 10:
    interesting_indices = [0, 2, 4, 5]
else:
    interesting_indices = [0, 2, 4, 5, 10, 11, 12, 13]

output_df, output_df_shap = pd.DataFrame(index = range(int(args.nreps_per_job) * int(len(interesting_indices))), columns = cols), pd.DataFrame(index = range(int(args.nreps_per_job) * int(len(interesting_indices))), columns = cols)

# replicate the simulation B times
start = time.time()
start_indx = 0
for b in range(1, int(args.nreps_per_job) + 1):
    # get all shapley values, mean(abs(SHAP values))
    shapley_vals, shapley_ics, shap_values, num_subsets_sampled, all_mps, p_values, hyp_tests = rs.do_one(n_train = current['n'], n_test = current['n'], p = p, m = m, measure_type = measure, binary = binary, gamma = gamma, cor = cor, V = args.V, estimator_type = args.est_type, conditional_mean = args.conditional_mean)
    # obtain flattened SPVIM vals and SHAP vals
    spvim_vals = shapley_vals[1:].flatten()
    mean_abs_shap = np.mean(np.absolute(shap_values), axis = 0)
    # obtain CIs for each SPVIM value
    spvim_lst, spvim_se_lst, spvim_ci_lst = zip(*(gsv.get_shapley_value(shapley_vals.flatten(), shapley_ics, j, level = 0.95, gamma = gamma) for j in range(1, p + 1)))
    spvim_ests = np.array(spvim_lst)
    spvim_ses = np.array(spvim_se_lst)
    spvim_cis = np.array(spvim_ci_lst)
    # add to dfs; first the SPVIM df
    end_indx = start_indx + int(len(interesting_indices))
    this_output = pd.DataFrame(index = range(start_indx, end_indx), columns = cols)
    this_output.loc[:, 'mc_id'] = b
    this_output.loc[:, 'measure'] = measure
    this_output.loc[:, 'n'] = current['n']
    this_output.loc[:, 's'] = [str(j) for j in interesting_indices]
    this_output.loc[:, 'est'] = spvim_ests[interesting_indices]
    this_output.loc[:, 'se'] = spvim_ses[interesting_indices]
    this_output.loc[:, 'cil'] = spvim_cis[interesting_indices, 0]
    this_output.loc[:, 'ciu'] = spvim_cis[interesting_indices, 1]
    this_output.loc[:, 'p_value'] = p_values.flatten()[interesting_indices]
    this_output.loc[:, 'hyp_test'] = hyp_tests.flatten()[interesting_indices]
    this_output.loc[:, 'num_subsets_sampled'] = num_subsets_sampled
    all_mp_array = np.tile(all_mps.reshape((all_mps.shape[0], 1)), len(interesting_indices)).transpose()
    aug_mp_array = np.empty((len(interesting_indices), len(mp_cols)))
    aug_mp_array[:] = np.nan
    aug_mp_array[:, :all_mp_array.shape[1]] = all_mp_array
    this_output.loc[:, mp_cols] = aug_mp_array
    # next the SHAP df
    this_shap_output = pd.DataFrame(index = range(start_indx, end_indx), columns = cols)
    this_shap_output.loc[:, 'mc_id'] = b
    this_shap_output.loc[:, 'measure'] = 'mean_abs_shap'
    this_shap_output.loc[:, 'n'] = current['n']
    this_shap_output.loc[:, 's'] = [str(j) for j in interesting_indices]
    this_shap_output.loc[:, 'est'] = mean_abs_shap[interesting_indices]
    # insert the current output into the main df
    output_df.iloc[start_indx:end_indx, :] = this_output
    output_df_shap.iloc[start_indx:end_indx, :] = this_shap_output
    start_indx = end_indx


end = time.time()
print(end - start)
# concatenate dfs together
output_df = pd.concat([output_df, output_df_shap], ignore_index = True)
# save it off
output_df.to_csv(args.sim_name + '_est_' + args.est_type + '_condmean_' + args.conditional_mean + '_output_' + str(job_id) + '.csv')
