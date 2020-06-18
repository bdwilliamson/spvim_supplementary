# Reproducing the MIMIC-II data analysis from the SPVIM paper

This file describes how to reproduce the MIMIC-II data analysis from ["Efficient nonparametric statistical inference on population feature importance using Shapley values"](https://arxiv.org/abs/2006.09481) by Williamson and Feng (*arXiv*, 2020; to appear in the Proceedings of the Thirty-seventh International Conference on Machine Learning [ICML 2020]). While the code in this file assumes that the user is submitting batch jobs to a high-performance computing (HPC) cluster using the Slurm batch scheduing system, minor edits to these commands allow the use of either local or alternative HPC cluster environments. All analyses were implemented in the freely available software packages Python and R; specifically, Python version 3.7.4 and R version 3.6.3.

The batch submission script `submit_all_icu_data_analysis.sh` can be used to reproduce all of the MIMIC-II data analyses. In addition to the results reported in the main manuscript --- assessing importance via the area under the receiver operating characteristic curve (AUC) and using neural networks and boosted trees to estimate the conditional means --- this script also runs analyses assessing importance via R-squared, for comparison. Running `./submit_all_icu_data_analysis.sh` will submit four batch submission jobs to a Slurm cluster.

The script calls `submit_icu_data_analysis.sh`, which must be edited to run on your particular environment. This submits an individual analysis for a chosen measure of predictiveness (R-squared or AUC) and a particular estimation method (trees or neural networks).

The script calls `icu_data_analysis.sh`, which performs the following two steps: (1) run `process_icu_data.py`, which processes the raw MIMIC-II data (located in the `icu_data/` directory) and creates a `.pkl` file with the results (this preprocessing step is described more fully in the Supplement); and (2) run `icu_data_analysis.py`, which runs the analysis and returns `.pkl` files and a `.csv` file with the estimated SPVIM values, SHAP values, LIME results, and conditional VIM values.

Once you have results from the analysis, run the code `reformat_icu_data.py` to create R-friendly data objects, and then run the code `load_icu_analysis.R` to generate plots with results.

The analyses make use of the following code:
* `utils.py`: general utility functions
* `compute_ic`: compute the influence function for a single predictiveness measure
* `measures_of_predictiveness.py`: predictiveness measures (e.g., R-squared, AUC) and functions for computing influence function estimates
* `shapley_hyp_test.py`: hypothesis test based on SPVIM estimates and standard errors
* `get_influence_functions.py`: compute influence functions for SPVIM values based on sampling and on the influence functions for estimated predictiveness measures
* `get_shapley_value.py`: return the shapley value, SE, CI, etc. for a given feature of interest
