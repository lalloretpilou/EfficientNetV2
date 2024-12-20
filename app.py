import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Charger le modèle pré-entraîné
MODEL_PATH = "EfficientNetV2B0V2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Classes des chiens
CLASSES = ['Chihuahua', 'Sussex_spaniel', 'Yorkshire_terrier', 'miniature_schnauzer']

IMG_SIZE = (224, 224)

# Fonction pour effectuer une prédiction
def predict(image):
    image = image.resize(IMG_SIZE)
    image_array = img_to_array(image) / 255.0  # Normaliser les pixels entre 0 et 1
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter une dimension pour le batch

    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index]

    return CLASSES[predicted_index], confidence, predictions[0]

# Interface utilisateur avec Streamlit
st.title("Prédiction de races de chiens")
st.write("Chargez une image de chien pour obtenir la prédiction.")

# Charger une image
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image chargée
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargée", use_container_width=True)  # Remplacement ici

    # Effectuer la prédiction
    predicted_race, confidence, probabilities = predict(image)

    # Afficher le résultat principal
    st.write(f"### Race prédite : {predicted_race}")
    st.write(f"### Confiance : {confidence:.2f}")

    # Afficher les probabilités pour toutes les classes
    st.write("### Probabilités pour chaque classe :")
    for i, class_name in enumerate(CLASSES):
        st.write(f"{class_name}: {probabilities[i] * 100:.2f}%")

graph_path = 'class_distribution.png'

st.header("Distribution des images par classe & Echantillon d'image par race:")
st.image(graph_path, caption="Distribution des images par classe", use_column_width=True)
