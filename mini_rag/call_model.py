import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# Charger embeddings et index
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.index")

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

def get_relevant_chunks(query, k=3):
    q_vec = model.encode([query])
    D, I = index.search(np.array(q_vec), k)
    return [chunks[i] for i in I[0]]

def ask_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    return response.json()["response"]

# query = "Quels sont les points clés du document ?"
# query = "A quelle adresse mail contacter les RH ?"
# query = "Si j'ai fait une visite médicale en 2025, dois-je en refaire une?"
query = "Comment remplir mon CRA?"
context = "\n\n".join(get_relevant_chunks(query))
prompt = f"Réponds à la question suivante en utilisant ce contexte :\n{context}\n\nQuestion : {query}"

answer = ask_ollama(prompt)
print(answer)
