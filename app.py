from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import requests

app = Flask(__name__)

# URL de téléchargement du modèle sur GitHub (fichier brut)
model_url = 'https://github.com/lalloretpilou/EfficientNetV2/blob/3b2e0b1af9af806e8173e092e75011f5dc81491b/model_transfer_learning_EfficientNetV2.h5'
model_path = '/tmp/model.h5'

if not os.path.exists(model_path):
    print("Téléchargement du modèle depuis GitHub...")
    r = requests.get(model_url, allow_redirects=True)
    open(model_path, 'wb').write(r.content)

model = tf.keras.models.load_model(model_path)

@app.route('/')
def home():
    return 'Bienvenue sur le modèle Heroku !'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data['input']).reshape(1, -1)
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)
    return jsonify({
        'prediction': predicted_class.tolist(),
        'probabilities': prediction.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
