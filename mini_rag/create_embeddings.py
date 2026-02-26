import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber


# Charger modèle embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Lire documents
texts = []
for filename in os.listdir("docs"):
    if filename.endswith(".pdf"):
        with pdfplumber.open(f"docs/{filename}") as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        texts.append(text)
# print(texts)

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text("\n".join(texts))

# Créer embeddings
embeddings = np.array([model.encode(chunk) for chunk in chunks])

# Créer index FAISS
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Sauvegarder index
faiss.write_index(index, "faiss_index.index")

# Sauvegarder chunks
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f)

print("Index et chunks sauvegardés.")
