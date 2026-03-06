from typing import TypedDict, List
from retriever import search
from rag_pipeline import rerank_chunks
from llm import ask_ollama
from langgraph.graph import StateGraph, START, END
import time


class GraphState(TypedDict):

    question: str
    chunks: List[str]
    context: str
    answer: str


def retrieve_node(state):

    start = time.time()
    question = state["question"]
    chunks = search(question)
    print("retrieve:", time.time() - start)

    return {
        "chunks": chunks
    }


def rerank_node(state):

    start = time.time()
    question = state["question"]
    chunks = state["chunks"]
    top_chunks = rerank_chunks(question, chunks, top_k=3)
    context = "\n\n".join(top_chunks)
    print("rerank:", time.time() - start)

    return {
        "context": context
    }


def llm_call(state):

    start = time.time()
    question = state["question"]
    context = state["context"]
    prompt = f"""
Réponds à la question en utilisant le contexte.

Contexte:
{context}

Question:
{question}
"""

    answer = ask_ollama(prompt)

    print("llm:", time.time() - start)
    return {
        "answer": answer
    }


builder = StateGraph(GraphState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("rerank", rerank_node)
builder.add_node("llm_call", llm_call)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "llm_call")
builder.add_edge("llm_call", END)

graph = builder.compile()
