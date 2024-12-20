import streamlit as st
import requests

# Titre de l'application
st.title("Catégorisez automatiquement des questions")

input_text = st.text_area("Entrez le contenu du post à taguer:", "")

api_url = "https://lalloret.pythonanywhere.com/predict"  # Remplacez par l'URL de votre API
model = tf.keras.models.load_model(api_url)

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
