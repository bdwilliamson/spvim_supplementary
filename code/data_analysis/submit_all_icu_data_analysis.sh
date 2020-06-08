#!/bin/bash

## submit all icu data analyses
./submit_icu_data_analysis.sh "auc" "tree"

./submit_icu_data_analysis.sh "r_squared" "tree"

./submit_icu_data_analysis.sh "auc" "nn"

./submit_icu_data_analysis.sh "r_squared" "nn"
