from backend.retriever import search
from backend.llm import ask_ollama


def rerank_chunks(question, chunks, top_k=3):
    return chunks[:top_k]
