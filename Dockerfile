FROM ubuntu:latest
MAINTAINER Bohdan Kovalevych bohdan.kovalevych@nure.ua

RUN apt-get -y update
RUN apt-get install -y git python-dev build-essential python3-pip python3

RUN git clone https://github.com/bkovalevych/big_data_service.git
WORKDIR ./big_data_service/app
RUN pip3 install -r requirements.txt

ENV MODEL_NAME=model/model.pkl
ENV DATASET_NAME=model/train.csv

RUN python3 model/init_script.py
CMD python3 app.py
