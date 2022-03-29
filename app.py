import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)
data_encoder = pickle.load(open('encoder.pkl', 'rb'))
model = pickle.load(open('pred_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    int_features = pd.DataFrame([int_features], columns=['Race_Name', 'Circuit_Name', 'Pit_Stop_Count', 'Pitted_On_Lap', 'Pit_Stop_Duration', 'Avg_Pit_Stop_Secs', 'Altitude_Metres', 'Fastest_Lap_Speed_KMH', 'Driver_Total_Points', 'Driver_Overall_Standing'])
    int_features["Race_Name"] = int_features["Race_Name"].astype(object)
    int_features["Circuit_Name"] = int_features["Circuit_Name"].astype(object)

    cat_cols = []
    num_cols = []
    for i in int_features.columns:
        if int_features[i].dtype == object:
            cat_cols.append(i)
        else:
            num_cols.append(i)
    encoded_cols = list(data_encoder.get_feature_names(cat_cols))
    int_features[encoded_cols] = data_encoder.transform(int_features[cat_cols])
    int_features = int_features[encoded_cols + num_cols]
    prediction = model.predict(int_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The average lap time is {} secs'.format(output))



if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=int(os.environ.get('PORT', 5000)))