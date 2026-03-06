import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
# import requests
from backend.ask import ask_question


with st.sidebar:
    st.header("About")
    st.markdown(
        """
        Ce chatbot est destiné à répondre aux questions concernant 5 Degrés.
        Il repose sur du RAG (Retrieval-Augmented Generation) sur des données
        internes telle que la documentation d'onboarding.
        """
    )

    st.header("Exemple de questions")
    st.markdown("- Comment remplir mon CRA?")
    st.markdown("- A quelle adresse mail contacter les RH ?")
    st.markdown("- Si j'ai fait une visite médicale en 2025, dois-je en refaire une?")

st.title("Chatbot 5D")

user_input = st.text_input("Pose ta question")

if st.button("Envoyer") and user_input:
    # response = requests.post(
    #     "http://localhost:8000/ask",
    #     json={"question": user_input}
    # )

    # if response.status_code == 200:
    #     answer = response.json()["answer"]
    #     st.write("### Réponse")
    #     st.write(answer)
    # else:
    #     st.error("Erreur API")

    with st.spinner("Recherche en cours..."):
        answer = ask_question(user_input)

    st.write("### Réponse")
    st.write(answer)
