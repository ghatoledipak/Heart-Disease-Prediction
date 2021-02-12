import numpy as np
import pickle
from flask import Flask, request, render_template

# Load ML model
modelgb = pickle.load(open('modelgb.pkl', 'rb'))
modelrf = pickle.load(open('modelrf.pkl', 'rb'))
modelknn = pickle.load(open('modelknn.pkl', 'rb'))

# Create application
app = Flask(__name__)

# Bind home function to URL


@app.route('/')
def home():
    return render_template('KNN.html')

# Bind predict function to URL


@app.route('/predictgb', methods=['POST'])
def predictgb():

    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = modelgb.predict(array_features)

    output = prediction

    # Check the output values and retrive the result with html tag based on the value
    if output == 1:
        return render_template('Heart Disease Classifier.html',
                               result='The patient is not likely to have heart disease!')
    else:
        return render_template('Heart Disease Classifier.html',
                               result='The patient is likely to have heart disease!')


@app.route('/predictrf', methods=['POST'])
def predictrf():

    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = modelrf.predict(array_features)

    output = prediction

    # Check the output values and retrive the result with html tag based on the value
    if output == 1:
        return render_template('Random_Forest.html',
                               result='The patient is not likely to have heart disease!')
    else:
        return render_template('Random_Forest.html',
                               result='The patient is likely to have heart disease!')


@app.route('/predictknn', methods=['POST'])
def predictknn():

    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = modelknn.predict(array_features)

    output = prediction

    # Check the output values and retrive the result with html tag based on the value
    if output == 1:
        return render_template('KNN.html',
                               result='The patient is not likely to have heart disease!')
    else:
        return render_template('KNN.html',
                               result='The patient is likely to have heart disease!')


if __name__ == '__main__':
    # Run the application
    app.run()
