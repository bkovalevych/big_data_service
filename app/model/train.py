#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import sys
import numpy as np
import pickle
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import xgboost
import warnings
warnings.filterwarnings("ignore")
pd.pandas.set_option("display.max_columns", None)
global_selected_feats = []


def train_model(data: pd.DataFrame):
    global global_selected_feats
    x_train = data.loc[:, ~data.columns.isin(['Unnamed: 0', 'price', 'zipcode'])]
    y_train = data["price"]

    sel_ = SelectFromModel(Lasso(alpha=0.005, random_state=1243))
    sel_.fit(x_train, y_train)
    selected_feats = x_train.columns[(sel_.estimator_.coef_ != 0).ravel().tolist()]
    for feature in selected_feats:
        global_selected_feats.append(feature)

    x_train = x_train[selected_feats]

    boost_model = xgboost.XGBRegressor(random_state=42, max_depth=13)
    boost_model.fit(x_train, y_train)
    return boost_model


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("There is no selected source name parameter which is 1-st parameter")
    else:
        file_name = sys.argv[1]
        result_name = "model_" + file_name[:-3] + "pkl"
        if sys.argv == 3:
            result_name = sys.argv[2]
        data_sample = pd.read_csv(file_name)
        result = train_model(data_sample)
        with open(result_name, "wb") as file:
            pickle.dump(result, file)
        print("Training model done!")
