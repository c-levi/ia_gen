from retriever import search
from llm import ask_ollama

def answer_question(question):

    context = "\n\n".join(search(question))

    prompt = f"""
Réponds à la question en te basant uniquement sur le contexte.

Contexte:
{context}

Question:
{question}
"""

    return ask_ollama(prompt)
