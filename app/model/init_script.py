import pandas as pd
import pickle
import os
import sys
sys.path.append(os.path.join(sys.path[0], 'model'))
from data_pre_processing import process_data
from train import train_model


model_name = os.environ["MODEL_NAME"]
dataset_name = os.environ["DATASET_NAME"]


train_data = pd.read_csv(dataset_name)
processed_train = process_data(train_data)
trained_model = train_model(processed_train)
with open(model_name, "wb") as file:
    pickle.dump(trained_model, file)


