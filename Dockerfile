FROM jupyter/scipy-notebook

COPY app/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN mkdir model
ENV MODEL_DIR=./model
ENV MODEL_FILE=model.pkl
ENV DATASET_FILE=train.csv

COPY app ./

RUN python3 train.py
