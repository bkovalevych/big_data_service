# FROM tiangolo/uwsgi-nginx-flask:python3.8
FROM python:3
MAINTAINER Bohdan Kovalevych bohdan.kovalevych@nure.ua

RUN apt-get -y update
RUN apt-get install -y git
# python-dev build-essential python3-pip python3
RUN git config --global http.sslverify "false"
RUN git clone https://github.com/bkovalevych/big_data_service.git
#RUN mv ./big_data_service/app/* /app
WORKDIR ./big_data_service/app
RUN pip3 install -r requirements.txt

ENV MODEL_NAME=model/model.pkl
ENV DATASET_NAME=model/train.csv

RUN python3 model/init_script.py

CMD python3 app.py
