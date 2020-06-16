# `spvim_supplementary`: Supplementary materials for the SPVIM paper

This repository contains the supplementary material for and code to reproduce the analyses in ["Efficient inference on population feature importance using Shapley values"](arXiv link here) by Williamson and Feng (*arXiv*, 2020). All analyses were implemented in the freely available software packages Python and R; specifically, Python version 3.7.4 and R version 3.6.3.

This README file provides an overview of the code available in the repository.

## Code directory

We have separated our code further into two sub-directories based on the two main objectives of the manuscript:

1. Numerical experiments to evaluate the operating characteristics of our proposed method (`sims`).
2. An analysis of patients' stays in the ICU from the Multiparameter Intelligent Monitoring in Intensive Care II ([MIMIC-II](https://mimic.physionet.org/)) database (`data_analysis`).

All analyses were performed on a Linux cluster using the Slurm batch scheduling system. The head node of the batch scheduler allows the shorthand "ml" in place of "module load". If you use a different batch scheduling system, the individual code files are flagged with the line where you can change batch variables. If you prefer to run the analyses locally, you may -- however, these analyses will then take a large amount of time.

-----

## Issues

If you encounter any bugs or have any specific questions about the analysis, please
[file an issue](https://github.com/bdwilliamson/spvim_supplementary/issues).
