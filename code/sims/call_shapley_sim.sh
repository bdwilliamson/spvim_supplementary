#!/bin/bash

## Command line args:
## 1: name of sim ("three-var" or "six-var")
## 2: number of total reps (e.g., 1000)
## 3: number of reps per job (e.g., 50)
## 4: estimator type ("tree" or "lm")
## 5: conditional mean ("nonlinear" or "linear")
python3 run_shapley_sim.py --sim-name $1 --nreps-total $2 --nreps-per-job $3 --est-type $4 --conditional-mean $5
