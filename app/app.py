import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import streamlit as st
import pickle
from src.data_preprocessing import preprocess_text

# Carregar modelo e vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def main():
    st.title("Assistente Inteligente para Gestão de Projetos de Infraestrutura")
    st.write("Digite o texto do projeto para classificação:")

    user_input = st.text_area("Texto do projeto", height=200)

    if st.button("Classificar"):
        if not user_input.strip():
            st.warning("Por favor, insira algum texto.")
            return

        # Pré-processar input
        processed_text = preprocess_text(user_input)

        # Carregar modelo e vectorizer
        model, vectorizer = load_model_and_vectorizer()

        # Transformar input
        X_input = vectorizer.transform([processed_text])

        # Fazer previsão
        prediction = model.predict(X_input)[0]

        st.success(f"Classificação prevista: **{prediction}**")

if __name__ == "__main__":
    main()
