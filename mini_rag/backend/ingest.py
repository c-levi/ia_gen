import os
import json
import faiss
import numpy as np
import pdfplumber

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCS_PATH = "data"
INDEX_PATH = "embeddings/faiss_index.index"
CHUNKS_PATH = "embeddings/chunks.json"


model = SentenceTransformer("all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

all_chunks = []
embeddings = []


for filename in os.listdir(DOCS_PATH):

    if not filename.endswith(".pdf"):
        continue

    filepath = os.path.join(DOCS_PATH, filename)

    with pdfplumber.open(filepath) as pdf:

        for page_number, page in enumerate(pdf.pages):

            text = page.extract_text()

            if not text:
                continue

            chunks = splitter.split_text(text)

            for chunk in chunks:

                chunk_data = {
                    "text": chunk,
                    "source": filename,
                    "page": page_number + 1
                }

                all_chunks.append(chunk_data)

                emb = model.encode(chunk)

                embeddings.append(emb)


embeddings = np.array(embeddings)


# FAISS index 

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)


# Save chunks 

with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)
