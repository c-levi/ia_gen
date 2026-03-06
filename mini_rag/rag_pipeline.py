from retriever import search
from llm import ask_ollama


def rerank_chunks(question, chunks, top_k=3):
    return chunks[:top_k]
