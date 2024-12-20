import streamlit as st
import requests

# Titre de l'application
st.title("Catégorisez automatiquement des questions")

input_text = st.text_area("Entrez le contenu du post à taguer:", "")

api_url = "https://github.com/lalloretpilou/EfficientNetV2/blob/f4a99658547fe1417fd31bba1825d0cc71516271/EfficientNetV2B0.h5"  # Remplacez par l'URL de votre API

# Fonction pour appeler l'API
def get_nlp_estimation(text):
    try:
        response = requests.post(api_url, json={"text": text})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Erreur lors de l'appel de l'API : {e}")
        return None

# Bouton pour effectuer l'estimation
if st.button("Analyser le texte"):
    if input_text:
        estimation = get_nlp_estimation(input_text)
        if estimation:
            st.subheader("Résultats de l'estimation")
            st.write(estimation)
    else:
        st.warning("Veuillez entrer un texte avant de lancer l'analyse.")
