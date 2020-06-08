#!/usr/local/bin/python3

## turn icu dataset into a csv for use in R

# required functions and libraries
import utils as uts
import numpy as np
import sys
import pandas as pd

# load the dataset
out = uts.pickle_from_file(sys.argv[1] + '/icu_data_processed.pkl')

# turn it into a csv
x_train, y_train, x_test, y_test = pd.DataFrame(data=out.x_train, index=np.array(range(1, out.x_train.shape[0] + 1)), columns=np.array(range(1, out.x_train.shape[1] + 1))), pd.DataFrame(data=out.y_train, index=np.array(range(1, out.y_train.shape[0] + 1)), columns=np.array(range(1, out.y_train.shape[1] + 1))), pd.DataFrame(data=out.x_test, index=np.array(range(1, out.x_test.shape[0] + 1)), columns=np.array(range(1, out.x_test.shape[1] + 1))), pd.DataFrame(data=out.y_test, index=np.array(range(1, out.y_test.shape[0] + 1)), columns=np.array(range(1, out.y_test.shape[1] + 1)))
x_train.to_csv(sys.argv[1] + '/icu_data_processed_xtrain.csv')
y_train.to_csv(sys.argv[1] + '/icu_data_processed_ytrain.csv')
x_test.to_csv(sys.argv[1] + '/icu_data_processed_xtest.csv')
y_test.to_csv(sys.argv[1] + '/icu_data_processed_ytest.csv')
