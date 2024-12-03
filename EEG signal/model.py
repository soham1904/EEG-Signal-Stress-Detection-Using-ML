from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle

app = Flask(__name__)

try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:

    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)


@app.route('/')
def home():
    return render_template('page.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('data')
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        data = np.array(data, dtype=float).reshape(1, -1)
        prediction = model.predict(data)[0]

        return jsonify({'prediction': str(prediction)})
    except ValueError as e:
        return jsonify({'error': f'Invalid input data: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

