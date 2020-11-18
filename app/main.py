from flask import Flask, request, render_template
import pandas as pd
import os
import sys
sys.path.append(os.path.join(sys.path[0], 'model'))
from app.model.data_pre_processing import process_data, cat_vars, features, dtypes, required_features
from app.model.predict import predict as predict_func
app = Flask(__name__)


@app.route('/')
def home():
    prepare_step = {}
    for key, value in dtypes.items():
        if value == 'float64':
            prepare_step[key] = "0.1"
        if value == 'int64':
            prepare_step[key] = "1"
    return render_template('index.html',
                           num_inputs=features,
                           cat_inputs=cat_vars,
                           steps=prepare_step,
                           required_features=required_features)


@app.route('/predict', methods=['POST'])
def predict():
    form = dict()
    savedForm = dict()
    for key, value in request.form.items():
        savedForm[key] = value
        prepared_value = value
        if value == '' and dtypes[key] != "O":
            prepared_value = float('NaN')
        if value == '' and dtypes[key] == "O":
            prepared_value = None
        form[key] = pd.Series([prepared_value], dtype=dtypes[key])
    data = pd.DataFrame(form, index=[0])
    prediction = round(predict_func(data), 2)
    prepare_step = {}
    for key, value in dtypes.items():
        if value == 'float64':
            prepare_step[key] = "0.1"
        if value == 'int64':
            prepare_step[key] = "1"
    return render_template('index.html',
                           form=savedForm,
                           num_inputs=features,
                           cat_inputs=cat_vars,
                           steps=prepare_step,
                           prediction_text='Price should be $ {:.2f}'.format(prediction),
                           required_features=required_features)


if __name__ == "__main__":
    print("__name__")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0',
            port=port)
