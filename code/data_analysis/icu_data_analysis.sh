#!/bin/bash
## Run the ICU data analysis

## args:
## 1: measure ('auc' or 'r_squared')
## 2: alpha (for Adam SGD)
## 3: precomputed-shapley-vals (did we already do most of the work, but need to restart at a given point?)


## -------------------------------------------------
## set up the data, read it in
## -------------------------------------------------
echo 'Processing data \n'
SECONDS=0
python process_icu_data.py 1 1 'icu_data/'
echo $SECONDS


## -------------------------------------------------
## estimate the shapley value!
## -------------------------------------------------
## using boosted trees
echo 'Estimating shapley values \n'
SECONDS=0
python icu_data_analysis.py --dataset 'icu_data/icu_data_processed.pkl' --seed 4747 --output-dir 'results/' --measure $1 --estimator-type $2
echo $SECONDS

## -------------------------------------------------
## create plots
## -------------------------------------------------
