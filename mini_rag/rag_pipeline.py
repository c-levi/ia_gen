from retriever import search
from llm import ask_ollama


def rerank_chunks(question, chunks, top_k=3):
    """
    Rerank des chunks par pertinence par rapport à la question.
    Retourne les top_k chunks les plus pertinents.
    """
    scored = []

    for chunk in chunks:
        prompt = f"""
Évalue la pertinence de ce texte par rapport à la question suivante.
Réponds uniquement par un score de 1 à 10 (10 = très pertinent).

Question:
{question}

Texte:
{chunk}
"""
        score_text = ask_ollama(prompt)
        try:
            score = int(score_text.strip())
        except:
            score = 0

        scored.append((score, chunk))

    # trier par score décroissant
    scored.sort(reverse=True, key=lambda x: x[0])

    # ne garder que les top_k
    top_chunks = [c for _, c in scored[:top_k]]

    return top_chunks


def answer_question(question):
    # récupération initiale
    chunks = search(question)

    # reranking pour ne garder que les 3 meilleurs
    top_chunks = rerank_chunks(question, chunks, top_k=3)

    context = "\n\n".join(top_chunks)

    prompt = f"""
Réponds à la question suivante en utilisant uniquement le contexte fourni.

Contexte:
{context}

Question:
{question}
"""

    return ask_ollama(prompt)
