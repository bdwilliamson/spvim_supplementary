import os
import sys
import csv
import json

import scipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

from utils import pickle_to_file
import data_generator


def _get_last_datapoint(df):
    return df.Value.values[-1]


def _get_mean(df):
    if df.Value.size == 1:
        return df.Value.values[0]
    elif np.unique(df.Time).size == 1:
        return df.Value.mean()
    mean_time = df.Time.mean()
    # lin_fit = scipy.stats.linregress(df.Time - mean_time, df.Value)
    x_init = df.Time - mean_time
    x = sm.add_constant(x_init)
    lin_fit = sm.OLS(df.Value, x, missing = 'drop').fit()
    return lin_fit.params[0]


def _get_max(df):
    return df.Value.max()


def _get_min(df):
    return df.Value.min()


def _get_sum(df):
    return df.Value.sum()


def _get_identity(x):
    return x.Value.values[0]


def _get_slope(df):
    if df.Value.size == 1 or np.unique(df.Time).size == 1:
        return 0
    return scipy.stats.linregress(df.Time / 50., df.Value)[0]


LAST = _get_last_datapoint
MIN = _get_min
MAX = _get_max
WEIGHTED_MEAN = _get_mean
SUM = _get_sum
IDENTITY = _get_identity
SLOPE = _get_slope


# FEATURES = {
#     # Based on the paper
#     "GCS": [SLOPE, LAST, WEIGHTED_MEAN, MAX, MIN],
#     "HCO3": [MIN, MAX, LAST, WEIGHTED_MEAN],
#     "BUN": [MIN, MAX, LAST, WEIGHTED_MEAN],
#     "Urine": [SUM],
#     "Age": [IDENTITY],
#     "SysABP": [WEIGHTED_MEAN, LAST, MIN, MAX],
#     "WBC": [LAST, WEIGHTED_MEAN, MIN, MAX],
#     "Temp": [WEIGHTED_MEAN, LAST, MIN, MAX],
#     "Glucose": [MAX, MIN, WEIGHTED_MEAN],
#     "Na": [WEIGHTED_MEAN, MAX, MIN],
#     "Lactate": [LAST, WEIGHTED_MEAN, MIN, MAX],
#     # Based on SAPS II or SAPS I (hhttps://archive.physionet.org/challenge/2012/saps_score.m)
#     "HR": [MIN, MAX, WEIGHTED_MEAN],
#     "K": [MIN, MAX, WEIGHTED_MEAN],
#     "ICUType": [IDENTITY],
#     "HCT": [WEIGHTED_MEAN, MIN, MAX],
#     "RespRate": [WEIGHTED_MEAN, MIN, MAX],
#     "MechVent": [MAX],
#     # Based on most common measurements
#     # "Creatinine": [WEIGHTED_MEAN, MIN, MAX],
#     # "Platelets": [WEIGHTED_MEAN, MIN, MAX],
#     # "Mg": [WEIGHTED_MEAN, MIN, MAX],
#     # Baseline measurements, general descriptors
#     "Gender": [IDENTITY],
#     "Weight": [IDENTITY],
#     "Height": [IDENTITY],
# }

# META_FEATURE_GROUPS = {
#         "GCS": ["GCS"],
#         "Metabolic": ["HCO3", "BUN", "Na", "K", "Glucose"],
#         "SysABP": ["SysABP"],
#         "CBC": ["WBC", "HCT"],
#         "Temp": ["Temp"],
#         "Lactate": ["Lactate"],
#         "HR": ["HR"],
#         "Respiration": ["RespRate", "MechVent", "O2"],
#         "Urine": ["Urine"],
#         "General Desc": ["Gender", "Height", "Weight", "Age", "ICUType"],
# }
META_FEATURE_GROUPS = {}

# Rather than get multiple features per group, only want one
# Choose them to cut down on missing data, if possible
FEATURES = {
    # Based on the paper
    "GCS": [MIN, WEIGHTED_MEAN, MAX],
    "HCO3": [MIN, WEIGHTED_MEAN, MAX],
    "BUN": [MIN, WEIGHTED_MEAN, MAX],
    "Urine": [MIN, WEIGHTED_MEAN, MAX],
    "Age": [IDENTITY],
    "SysABP": [MIN, WEIGHTED_MEAN, MAX],
    "WBC": [MIN, WEIGHTED_MEAN, MAX],
    "Temp": [MIN, WEIGHTED_MEAN, MAX],
    "Glucose": [MIN, WEIGHTED_MEAN, MAX],
    "Na": [MIN, WEIGHTED_MEAN, MAX],
    "Lactate": [MIN, WEIGHTED_MEAN, MAX],
    "HR": [MIN, WEIGHTED_MEAN, MAX],
    "K": [MIN, WEIGHTED_MEAN, MAX],
    "ICUType": [IDENTITY],
    "HCT": [MIN, WEIGHTED_MEAN, MAX],
    "RespRate": [MIN, WEIGHTED_MEAN, MAX],
    "MechVent": [MIN, WEIGHTED_MEAN, MAX], # this is a flag if they were on mechanical ventilation
    # Baseline measurements, general descriptors
    "Gender": [IDENTITY],
    "Weight": [IDENTITY],
    "Height": [IDENTITY],
}

NORMAL_RANGES = {
    "GCS": [15, 15],
    "HCO3": [20, 30],
    "BUN": [8, 28],
    "Urine": [2000, 4000],
    "SysABP": [100, 199],
    "WBC": [1, 19.9],
    "Temp": [36, 38.9],
    "Glucose": [62, 125],
    "Na": [135, 145],
    "Lactate": [0.5, 1],
    "HR": [70, 119],
    "K": [3.6, 5.2],
    "HCT": [36, 45],
    "RespRate": [12, 20],
    "MechVent": [0, 0],
    "O2": [200, 250],
}

MAX_PROCESS = 5000


def _process_feature_groups(col_names):
    """
    @return List of feature indices that correspond to a single group that we want to measure importance of
            Dictionary mapping each of these groups to their name and whether or not they are "individual" variable groups of "meta"-groups
            Dictionary mapping variable to normal ranges and the indices of features extracted from that variable
    """
    feature_groups = {}
    for feature_idx, col_name in enumerate(col_names):
        measure, processing_func_name = col_name.split(":")
        measure = measure.strip()
        if measure not in feature_groups:
            feature_groups[measure] = {processing_func_name: feature_idx}
        else:
            feature_groups[measure][processing_func_name] = feature_idx
    print(len(feature_groups))
    # Create nan fill config
    nan_fill_config = {}
    for measure, feature_dict in feature_groups.items():
        if measure in NORMAL_RANGES:
            nan_fill_config[measure] = {
                    "indices": list(feature_dict.values()),
                    "range": NORMAL_RANGES[measure]}
    # Create dictionary mapping variable importance group idx to the group name and some bool flags
    feature_group_list = []
    measure_names = {}
    vi_idx = 0
    for measure, feature_dict in feature_groups.items():
        measure_names[vi_idx] = {
                "name": "%s" % measure,
                "is_ind": 1,
                "is_group": int(measure in META_FEATURE_GROUPS)}
        print(measure, feature_dict.values())
        feature_group_list.append(",".join([str(i) for i in feature_dict.values()]))
        vi_idx += 1
    # Also process the meta groups
    for group_name, group_members in META_FEATURE_GROUPS.items():
        if len(group_members) > 1:
            measure_names[vi_idx] = {
                "name": group_name,
                "is_ind": 0,
                "is_group": 1}
            feature_idx_list = []
            for measure_name in group_members:
                feature_idx_list += list(feature_groups[measure_name].values())
            print(group_name, feature_idx_list)
            feature_group_list.append(",".join([str(i) for i in feature_idx_list]))
            vi_idx += 1
    assert len(feature_group_list) == len(measure_names)
    return feature_group_list, measure_names, nan_fill_config


def main(args=sys.argv[1:]):
    train_size = float(args[0])
    seed = int(args[1])
    icu_data_dir = args[2]

    # Read the y data
    outcomes = pd.read_csv(icu_data_dir + "Outcomes-a.txt")
    subject_outcomes = outcomes[["RecordID", "In-hospital_death"]]

    # Create a dictionary of features for each subject
    # Using a dictionary because some of the features don't appear in all subjects...
    value_range = {}  # this is just for printing out ranges of the values
    file_folder = icu_data_dir + "set-a/"
    all_subject_features = {}
    for idx, filename in enumerate(os.listdir(file_folder)[:MAX_PROCESS]):
        df = pd.read_csv("%s%s" % (file_folder, filename))
        df["hour"] = np.array([time.split(":")[0] for time in df.Time.values], dtype=int)
        df["minute"] = np.array([time.split(":")[1] for time in df.Time.values], dtype=int)
        df.Time = df.hour * 60 + df.minute

        record_id = int(df.loc[0].Value)
        subject_features = {"RecordID": record_id}
        for feat_name, process_func_list in FEATURES.items():
            if WEIGHTED_MEAN in process_func_list:
                sub_df = df.loc[(df.Parameter == feat_name) & (df.Value > 0)]
            else:
                sub_df = df.loc[(df.Parameter == feat_name) & (df.Value >= 0)]

            if sub_df.shape[0] == 0:
                continue
            if feat_name not in value_range:
                value_range[feat_name] = [sub_df.Value.min(), sub_df.Value.max()]
            else:
                value_range[feat_name][0] = min(value_range[feat_name][0], sub_df.Value.min())
                value_range[feat_name][1] = max(value_range[feat_name][1], sub_df.Value.max())

            for func in process_func_list:
                value = func(sub_df)
                if not np.isfinite(value):
                    print (value, feat_name, func.__name__)
                    print (sub_df)
                assert np.isfinite(value)
                full_feature_name = "%s:%s" % (feat_name, func.__name__)
                subject_features[full_feature_name] = value

        fio2_df = df.loc[df.Parameter == "FiO2"]
        pao2_df = df.loc[df.Parameter == "PaO2"]
        if fio2_df.shape[0] and pao2_df.shape[0]:
            fio2_mean = _get_mean(fio2_df)
            pao2_mean = _get_mean(pao2_df)
            if fio2_mean > 0:
                subject_features["O2:_get_ratio"] = pao2_mean / fio2_mean

        all_subject_features[idx] = subject_features

    for k, v in value_range.items():
        print (k, v)

    subjects_x = pd.DataFrame.from_dict(all_subject_features, orient="index")

    ## if a covariate has > 30% missing data, remove it
    prop_nan = subjects_x.apply(lambda x: np.mean(np.isnan(x)))
    print('Features filtered for proportion of NA values >= 0.3')
    print(prop_nan >= 0.3)
    tmp = subjects_x.loc[:, prop_nan < 0.3]
    subjects_x = tmp

    # Merge the X and Y data
    icu_subjects = subjects_x.merge(subject_outcomes, on="RecordID")
    death_resp = icu_subjects["In-hospital_death"]
    icu_subjects = icu_subjects.drop(columns=["RecordID"])

    # Grab column names
    column_names = list(icu_subjects.columns.values)
    print(column_names)
    # icu_subjects = icu_subjects.as_matrix()
    icu_subjects = icu_subjects.loc[:, column_names].values

    # Center the x covariates
    centering_term = np.nanmean(icu_subjects, axis=0)
    centering_term[-1] = 0
    icu_subjects -= centering_term
    assert np.all(death_resp == icu_subjects[:, -1])

    # randomly split the data
    if train_size < 1:
        mats = train_test_split(icu_subjects, train_size = train_size, test_size = 1.0 - train_size, random_state = seed)
        x_train = mats[0][:, :-1]
        y_train = mats[0][:, -1:]
        x_test = mats[1][:, :-1]
        y_test = mats[1][:, -1:]
    else:
        x_train = icu_subjects[:, :-1]
        y_train = icu_subjects[:, -1:]
        x_test = x_train
        y_test = y_train

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # Save the data
    icu_data = data_generator.Dataset(
        x_train = x_train,
        y_train = y_train,
        x_test = x_test,
        y_test = y_test)

    ## save off as a pickle
    icu_processed_file = icu_data_dir + "icu_data_processed.pkl"
    pickle_to_file(icu_data, icu_processed_file)

    icu_column_file = icu_data_dir + "icu_data_column_names.txt"
    with open(icu_column_file, "w") as f:
        for i, col in enumerate(column_names[:-1]):
            f.write("%d, %s\n" % (i, col))

    feature_group_list, vi_group_names, nan_fill_config = _process_feature_groups(column_names[:-1])
    print("Copy paste this for creating the variable importance groups argument!")
    print("--var-import-idx %s" % ";".join(feature_group_list))
    icu_vi_name_file = icu_data_dir + "icu_data_var_import_names.csv"
    vi_group_name_df = pd.DataFrame.from_dict(vi_group_names, orient="index")
    vi_group_name_df.to_csv(icu_vi_name_file)

    nan_config_file = icu_data_dir + "nan_fill_config.json"
    with open(nan_config_file, 'w') as f:
        json.dump(nan_fill_config, f)

if __name__ == "__main__":
    main(sys.argv[1:])
