from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hours = float(request.form['hours'])
    prediction = model.predict([[hours]])

    return render_template('index.html', 
                           prediction_text="Predicted Score: " + str(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)