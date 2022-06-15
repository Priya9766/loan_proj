import pandas as pd
import numpy as np
from flask import Flask,request,render_template
import pickle




with open('scaler.pkl','rb') as scaler_file:
    scaler=pickle.load(scaler_file)

with open('model.pkl','rb') as model_file:
    model=pickle.load(model_file)

app=Flask(__name__)

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/data',methods=['GET','POST'])
def data():

    user_data = request.form
    print(user_data)

    user_age = request.form['age']
    user_annual = request.form['annual']
    user_score = request.form['score']

    
    scale_data = scaler.transform([[user_age,user_annual,user_score]])
    result = model.predict(scale_data)
    print(result)
    
    return render_template('display.html',data=result)


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080,debug=False)