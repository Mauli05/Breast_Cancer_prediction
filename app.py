from re import X
from flask import Flask,render_template,request,url_for
import pickle
import numpy as np
import pandas as pd


# create an object of the class "Flask" by passing first argument.
app = Flask(__name__)
model = pickle.load(open("B_cancer_model.pkl", "rb"))
app = Flask(__name__,template_folder='templates')
# app = Flask(__name__,static_folder='Statics')

@app.route("/")
def home():
    return render_template("index1.html")

# @app.route("/predict", methods = ["POST"]) 
# def predict():
#     input_features = [float(X) for x in request.form.values()]
#     features_value = [np.array(input_features)]
#     features_name = ['texture_mean', 'area_mean',
#        'smoothness_mean', 'concavity_mean','texture_se','area_se','texture_worst',
#        'smoothness_worst','compactness_worst', 'symmetry_worst']

#     #df = pd.DataFrame(features_value, columns=features_name)
#     output = model.predict(features_value) 
#     print("op check",output)  
#     if output==1:
#         return "<h1 style='color:rgb(39, 2, 75);'> <u> type malignant cancer </u></h1>"
#     else:
#         return "<h2 style='color:rgb(39, 2, 75);'> <u>  NO cancer to this petient </u></h2>"
    #return render_template("after.html", data=output)
    
@app.route("/predict",methods=["POST"])
def predict():
    texture_mean=float(request.form.get("texture_mean"))
    area_mean=float(request.form.get("area_mean"))
    smoothness_mean=float(request.form.get("smoothness_mean"))
    concavity_mean=float(request.form.get("concavity_mean"))
    texture_se=float(request.form.get("texture_se"))
    area_se=float(request.form.get("area_se"))
    texture_worst=float(request.form.get("texture_worst"))
    smoothness_worst=float(request.form.get("smoothness_worst"))
    compactness_worst=float(request.form.get("compactness_worst"))
    symmetry_worst=float(request.form.get("symmetry_worst"))
    
    
    result=model.predict(np.array([texture_mean, area_mean,
       smoothness_mean, concavity_mean,texture_se,area_se,texture_worst,
       smoothness_worst,compactness_worst, symmetry_worst]).reshape(1,10))
    
    if result==1:
        return "<h1 style='color:green'> patient has cancer</h1>"
    else:
        return "<h1 style='color:red'>patient has No cancer</h1>"
    # prediction =model.predict(features_value)
    # output = prediction[0]
    # return render_template('index1.html', prediction_text='cancer = {}'.format(output))
    # if output==1:
    #     return "<h1 style='color:rgb(39, 2, 75);'> <u> type malignant cancer </u></h1>"
    # else:  
    #     return "<h2 style='color:rgb(39, 2, 75);'> <u>  NO cancer to this petient </u></h2>"
    # return render_template("after.html", data=output)
    

