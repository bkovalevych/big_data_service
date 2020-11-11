import pickle
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(sys.path[0], 'model'))
from data_pre_processing import prepare_test


model_name = os.environ["MODEL_NAME"]

with open(model_name, "rb") as file:
    modelExec = pickle.load(file)


def predict(raw_data: pd.DataFrame):

    processed_data = prepare_test(raw_data)
    prediction = np.exp(modelExec.predict(processed_data))
    return prediction[0]

