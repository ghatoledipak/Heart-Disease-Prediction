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
    return render_template('index.html')

# Bind predict function to URL


@app.route('/showknn')
def showknn():
    return render_template('KNN.html')


@app.route('/showgb')
def showgb():
    return render_template('Heart Disease Classifier.html')


@app.route('/showrf')
def showrf():
    return render_template('Random_Forest.html')


@app.route('/predictgb', methods=['POST'])
def predictgb():

    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = modelgb.predict(array_features)

    x = modelgb.predict_proba(array_features)
    pos = x[0][1]
    pos = pos*100

    # neg = x[0][0]

    output = prediction

    # Check the output values and retrive the result with html tag based on the value
    if pos > 70:
        return render_template('Heart Disease Classifier.html',
                               result='Probablity of having heart disease: ', positive=pos, res2='Risk is HIGH')
    if pos > 40:
        return render_template('Heart Disease Classifier.html',
                               result='Probablity of having heart disease: ', positive=pos, res2='Risk is MEDIUM')
    else:
        return render_template('Heart Disease Classifier.html',
                               result='Probablity of having heart disease: ', positive=pos, res2='Risk is LOW')


@app.route('/predictrf', methods=['POST'])
def predictrf():

    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = modelrf.predict(array_features)

    output = prediction

    x = modelrf.predict_proba(array_features)
    pos = x[0][1]
    pos = pos*100

    # neg = x[0][0]

    output = prediction

    # Check the output values and retrive the result with html tag based on the value
    if pos > 70:
        return render_template('Random_Forest.html',
                               result='Probablity of having heart disease: ', positive=pos, res2='Risk is HIGH')
    if pos > 40:
        return render_template('Random_Forest.html',
                               result='Probablity of having heart disease: ', positive=pos, res2='Risk is MEDIUM')
    else:
        return render_template('Random_Forest.html',
                               result='Probablity of having heart disease: ', positive=pos, res2='Risk is LOW')


@app.route('/predictknn', methods=['POST'])
def predictknn():

    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = modelknn.predict(array_features)

    output = prediction

    x = modelknn.predict_proba(array_features)
    pos = x[0][1]
    pos = pos*100

    # neg = x[0][0]

    output = prediction

    # Check the output values and retrive the result with html tag based on the value
    if pos > 70:
        return render_template('KNN.html',
                               result='Probablity of having heart disease: ', positive=pos, res2='Risk is HIGH')
    if pos > 40:
        return render_template('KNN.html',
                               result='Probablity of having heart disease: ', positive=pos, res2='Risk is MEDIUM')
    else:
        return render_template('KNN.html',
                               result='Probablity of having heart disease: ', positive=pos, res2='Risk is LOW')


if __name__ == '__main__':
    # Run the application
    app.run()
