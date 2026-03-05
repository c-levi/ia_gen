import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# Charger embeddings et index
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.index")

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

def search(query, k=3):
    q_vec = model.encode([query])
    D, I = index.search(np.array(q_vec), k)
    return [chunks[i] for i in I[0]]
