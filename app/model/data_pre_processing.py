#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
from .train import global_selected_feats as selected_feats

pd.pandas.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

features = []
cat_vars = dict()
dtypes = dict()
required_features = dict()

frequent_labels = dict()
ordinal_labels = dict()
max_year = 2000
scaler = None
train_vars = None
mode_vals = dict()

def add_columns(df, is_test=False):
    global max_year
    max_val = df['registration_year'].max()
    if is_test:
        max_val = max_year
    else:
        max_year = max_val
    df['years_old'] = max_val - df['registration_year']
    df['is_old'] = np.where(df['registration_year'] < 1980, 1, 0)
    df['is_new'] = np.where(df['registration_year'] + 10 > max_val, 1, 0)


def find_frequent_labels(df, var, rare_perc):
    df = df.copy()
    tmp = df.groupby(var)['price'].count() / len(df)
    return tmp[tmp > rare_perc].index


def add_model(df, categories):
    global cat_vars
    cat_vars = dict()
    for category in categories:
        if df[category].dtypes == 'O':
            cat_vars[category] = []
            if not required_features[category]:
                cat_vars[category].append("")
            for var in df[category].unique():
                cat_vars[category].append(str(var))
                df[category + "_" + str(var)] = np.where(df[category] == var, 1, 0)
        else:
            for var in df[category].unique():
                df[category + "_" + str(var)] = np.where(df[category] == var, 1, 0)


def replace_categories(train, var, target):
    global ordinal_labels
    ordered_labels = train.groupby([var])[target].mean().sort_values().index
    ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}
    ordinal_labels[var] = ordinal_label
    train[var] = train[var].map(ordinal_label)


def base_process(data: pd.DataFrame):
    vars_with_na = [var for var in data if data[var].isnull().sum() > 0 and data[var].dtypes == 'O']

    data[vars_with_na] = data[vars_with_na].fillna("Missing")

    num_na = [var for var in data if data[var].isnull().sum() > 0 and data[var].dtypes != 'O']
    for var in num_na:
        mode_val = data[var].mode()[0]
        mode_vals[var] = mode_val
        data[var + '_na'] = np.where(data[var].isnull(), 1, 0)

        data[var] = data[var].fillna(mode_val)


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    global features, dtypes, frequent_labels, ordinal_labels, scaler, train_vars, required_features
    features = [var for var in data.columns if data[var].dtypes != 'O' and var not in ["Unnamed: 0", "price"]]
    for var in data.columns:
        if data[var].isnull().sum() > 0:
            required_features[var] = False
        else:
            required_features[var] = True

    dtypes = dict(data.dtypes)

    base_process(data)
    add_columns(data)
    cat_variables = [var for var in data.columns if data[var].dtype == 'O']
    add_model(data, ["mileage"] + cat_variables)
    frequent_labels = dict()
    for var in cat_variables:
        frequent_ls = find_frequent_labels(data, var, 0.03)
        frequent_labels[var] = frequent_ls
        data[var] = np.where(data[var].isin(
            frequent_ls), data[var], 'Rare')

    ordinal_labels = dict()
    for var in cat_vars:
        replace_categories(data, var, 'price')

    data['price'] = np.log(data['price'])
    train_vars = [var for var in data.columns if var not in ["Unnamed: 0", 'price', 'zipcode']]
    data['insurance_price'] = np.log(data['insurance_price'])
    scaler = MinMaxScaler()
    scaler.fit(data[train_vars])

    data[train_vars] = scaler.transform(data[train_vars])

    return data


def prepare_test(data: pd.DataFrame):
    vars_with_na = [var for var in data if data[var].isnull().sum() > 0 and data[var].dtypes == 'O']
    data[vars_with_na] = data[vars_with_na].fillna("Missing")
    num_na = [var for var in data if data[var].isnull().sum() > 0 and data[var].dtypes != 'O']
    for var in num_na:
        mode_val = mode_vals[var]
        data[var + '_na'] = np.where(data[var].isnull(), 1, 0)
        data[var] = data[var].fillna(mode_val)
    add_columns(data, True)
    for var in cat_vars:
        frequent_ls = frequent_labels[var]
        data[var] = np.where(data[var].isin(
            frequent_ls), data[var], 'Rare')

    for var in cat_vars:
        data[var] = data[var].map(ordinal_labels[var])
    for var in train_vars:
        if data.get(var) is None:
            data[var] = 0
    data['insurance_price'] = np.log(data['insurance_price'])
    data[train_vars] = scaler.transform(data[train_vars])
    return data[selected_feats]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("There is no selected source name parameter which is 1-st parameter")
    else:
        file_name = sys.argv[1]
        result_name = "preprocessed_" + file_name
        if sys.argv == 3:
            result_name = sys.argv[2]
        data_sample = pd.read_csv(file_name)
        result = process_data(data_sample)
        result.to_csv(result_name, index=False)
        print("done")

