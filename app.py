import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

def load_model():
    return pickle.load(open('loan_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    labels = ['Rejected', 'Accepted']

    extract = [float(x) for x in request.form.values()]
    features = []
    for i in range(len(extract)-2):
        features.append(extract[i])
    if extract[9] == 0.0:
        features.append(1)
        features.append(0)
        features.append(0)
    if extract[9] == 1.0:
        features.append(0)
        features.append(1)
        features.append(0)
    if extract[9] == 2.0:
        features.append(0)
        features.append(0)
        features.append(1)
    if extract[10] == 0.0:
        features.append(1)
        features.append(0)
        features.append(0)
        features.append(0)
    if extract[10] == 1.0:
        features.append(0)
        features.append(1)
        features.append(0)
        features.append(0)
    if extract[10] == 2.0:
        features.append(0)
        features.append(0)
        features.append(1)
        features.append(0)
    if extract[10] == 3.0:
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(1)
    values = [np.array(features)]

    model = load_model()
    prediction = model.predict(values)
    result = labels[prediction[0]]

    return render_template('index.html', output= 'Your Loan has been {}'.format(result))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True)