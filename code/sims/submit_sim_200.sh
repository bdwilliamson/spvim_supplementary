#!/bin/bash

## Command line args:
## 1: name of sim ("three-var" or "six-var")
## 2: number of total reps (e.g., 1000)
## 3: number of reps per job (e.g., 50)

./submit_shapley_sim.sh "continuous-twohundred" 1000 5 6 "twohundred-var-output" "tree" "nonlinear"
