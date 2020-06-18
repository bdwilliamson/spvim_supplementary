# Running the numerical experiments for the SPVIM paper

This file describes how to reproduce the simulations in ["Efficient nonparametric statistical inference on population feature importance using Shapley values"](https://arxiv.org/abs/2006.09481) by Williamson and Feng (*arXiv*, 2020; to appear in the Proceedings of the Thirty-seventh International Conference on Machine Learning [ICML 2020]). While the code in this file assumes that the user is submitting batch jobs to a high-performance computing (HPC) cluster using the Slurm batch scheduing system, minor edits to these commands allow the use of either local or alternative HPC cluster environments. All analyses were implemented in the freely available software packages Python and R; specifically, Python version 3.7.4 and R version 3.6.3.

The batch submission script `submit_sim_200.sh` can be used to reproduce the simulations. Running `./submit_sim_200.sh` will submit a single batch submission job array to a Slurm cluster.

The batch submission script calls the `call_shapley_sim.sh` script, which runs the code `run_shapley_sim.py` with specified options. The python scripts require a job ID to pick off the correct simulation parameters for the specified run --- this job ID either comes from the batch submission scheduler or must be edited to take a user-specified value. Then, the python scripts set up files to write results to, and runs `run_shapley_sim_once.py` multiple times (by default, for 5 replications for a given set of parameters).

`run_shapley_sim_once.py` performs one replication of the simulation for a specified set of parameters: generate data (`data_generator.py` and `data_gen_funcs.py`) for the specified setting; estimate the SPVIM values; do hypothesis tests; estimate the SHAP values; return the results.

Once all simulations have finished, use `load_sim_shapley.R` to compile the results and produce plots. The file `true_shapley_vals_with_7_nonnoise_vars.rds` contains the true values of R-squared-based SPVIM for the seven non-null features considered in the simulations.

The analyses make use of the following code:
* `utils.py`: general utility functions
* `compute_ic`: compute the influence function for a single predictiveness measure
* `measures_of_predictiveness.py`: predictiveness measures (e.g., R-squared, AUC) and functions for computing influence function estimates
* `shapley_hyp_test.py`: hypothesis test based on SPVIM estimates and standard errors
* `get_influence_functions.py`: compute influence functions for SPVIM values based on sampling and on the influence functions for estimated predictiveness measures
* `get_shapley_value.py`: return the shapley value, SE, CI, etc. for a given feature of interest
* `sim_truth_utils.R`, `true_vals_200.R`: approximate the true SPVIM values using Monte-Carlo sampling
