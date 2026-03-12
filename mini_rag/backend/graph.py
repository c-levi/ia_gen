from typing import TypedDict, List
from backend.retriever import search
from backend.rag_pipeline import rerank_chunks
from backend.llm import ask_ollama
from langgraph.graph import StateGraph, START, END
import time


class GraphState(TypedDict):

    question: str
    history: List[str]
    rewritten_question: str
    chunks: List[str]
    context: str
    compressed_context: str
    answer: str


def rewrite_node(state):

    start = time.time()

    question = state["question"]
    history = "\n".join(state.get("history", []))

    prompt = f"""
Reformules la question utilisateur pour qu'elle soit autonome.

Si la question est déjà claire, renvoie-la telle quelle.

Historique:
{history}

Question:
{question}

Question reformulée:
"""

    rewritten = ask_ollama(prompt)

    print("rewrite:", time.time() - start)
    return {
        "rewritten_question": rewritten.strip()
    }


def retrieve_node(state):

    start = time.time()
    question = state["rewritten_question"]
    chunks = search(question)
    print("retrieve:", time.time() - start)

    return {
        "chunks": chunks
    }


def rerank_node(state):

    start = time.time()
    question = state["rewritten_question"]
    chunks = state["chunks"]
    top_chunks = rerank_chunks(question, chunks, top_k=3)
    context = "\n\n".join(top_chunks)
    print("rerank:", time.time() - start)

    return {
        "context": context
    }


def compress_node(state):

    start = time.time()
    question = state["rewritten_question"]
    context = state["context"]

    prompt = f"""
Extrais uniquement les informations utiles pour répondre à la question.

Contexte:
{context}

Question:
{question}

Informations pertinentes:
"""

    compressed = ask_ollama(prompt)

    print("compress:", time.time() - start)
    return {
        "compressed_context": compressed
    }


def llm_call(state):

    start = time.time()
    question = state["question"]
    context = state["compressed_context"]
    history = state.get("history", [])
    history_text = "\n".join(history)

    prompt = f"""
Historique de conversation:
{history_text}

Réponds à la question en utilisant le contexte.

Contexte:
{context}

Question:
{question}
"""

    answer = ask_ollama(prompt)

    print("llm:", time.time() - start)

    # Append to history
    history.append(f"User: {question}")
    history.append(f"Assistant: {answer}")

    return {
        "answer": answer,
        "history": history
    }


builder = StateGraph(GraphState)

builder.add_node("rewrite", rewrite_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("rerank", rerank_node)
builder.add_node("compress", compress_node)
builder.add_node("llm_call", llm_call)

builder.add_edge(START, "rewrite")
builder.add_edge("rewrite", "retrieve")
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "compress")
builder.add_edge("compress", "llm_call")
builder.add_edge("llm_call", END)

graph = builder.compile()
