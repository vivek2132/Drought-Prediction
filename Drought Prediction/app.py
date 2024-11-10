from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    population = float(data['population'])
    rain = float(data['rain'])

    features = np.array([[population, rain]])
    
    prediction = model.predict(features)

    result = 'Drought' if prediction[0] else 'No Drought'
    return render_template('index.html', prediction_text=f'The model predicts: {result}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    population = data['population']
    rain = data['rain']

    features = np.array([[population, rain]])

    prediction = model.predict(features)

    result = {'prediction': 'Drought' if prediction[0] else 'No Drought'}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


