#!/bin/bash

ml Python/3.7.4-foss-2019b-fh1

## export virtualenv to path
export PYTHONPATH=~/shapley/py3env/lib/python3.7/site-packages:$PYTHONPATH

## args
## 1: measure ('auc' or 'r_squared')
## 2: estimator type ('tree' or 'nn')

sbatch -c10 --mem 100G --time=7-0 ./icu_data_analysis.sh $1 $2
