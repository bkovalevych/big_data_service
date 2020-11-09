FROM ubuntu
MAINTAINER Bohdan Kovalevych bohdan.kovalevych@nure.ua

RUN apt-get update
RUN apt-get install -y git python-virtualenv
RUN git clone https://github.com/bkovalevych/big_data_service.git
RUN cd ./app
WORKDIR ./app
FROM jupyter/scipy-notebook


RUN pip install -r requirements.txt
ENV MODEL_NAME=model/model.pkl
ENV DATASET_NAME=model/train.csv

RUN python3 app.py
