import pickle

from django.shortcuts import render, redirect

import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('model1.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['GET','POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    final = [np.array(input_features)]
    #print(input_features)
    print(final)

    features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                     'mean smoothness']

    output = model.predict(final)

    if output == 0:
        res_val = "** breast cancer **"
        return render_template("index.html", pred='Patient has {}'.format(res_val))

    else:
        res_val = "no breast cancer"
        return render_template("index.html", pred='Patient has {}'.format(res_val))


if __name__ == "__main__":
    app.run(debug=True)
print('Hello')
