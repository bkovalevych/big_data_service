import pickle
import pandas as pd
import os
import numpy as np
from .data_pre_processing import prepare_test


model_name = os.environ["MODEL_NAME"]

with open(model_name, "rb") as file:
    modelExec = pickle.load(file)


def predict(raw_data: pd.DataFrame):

    processed_data = prepare_test(raw_data)
    prediction = np.exp(modelExec.predict(processed_data))
    return prediction[0]

