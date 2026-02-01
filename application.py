print("🔥 RUNNING application.py FILE 🔥")

import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

#import ridge regresiion and stamdard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler= pickle.load(open('models/scaler.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET',"POST"])
def predict_datapoint():
    if request.method=="POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        input_df = pd.DataFrame([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]],
                                columns=['Temperature','RH','Ws','Rain','FFMC','DMC','ISI','Classes','Region'])

        new_data_scaled = standard_scaler.transform(input_df)
        pred = ridge_model.predict(new_data_scaled)[0]

        return render_template('home.html', result=round(float(pred), 2))


    else:
        return render_template('home.html',result=None)

@app.route('/test')
def test():
    return "TEST OK"

if __name__== "__main__" :
    app.run(host="0.0.0.0")
