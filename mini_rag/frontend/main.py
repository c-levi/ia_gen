import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
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

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

if user_input := st.chat_input("Pose ta question"):
    st.chat_message("user").markdown(user_input)

    st.session_state.messages.append({"role": "user", "output": user_input})

    data = {"text": user_input}

    with st.spinner("Searching for an answer..."):
        history = [
            f"{m['role']}: {m['output']}"
            for m in st.session_state.messages
        ]
        answer = ask_question(user_input, history)

    st.chat_message("assistant").markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": answer
        }
    )
