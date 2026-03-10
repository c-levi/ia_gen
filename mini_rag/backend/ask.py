# from rag_pipeline import answer_question
from graph import graph


def ask_question(query: str, history):
    result = graph.invoke({
        "question": query,
        "history": history
    })
    return result["answer"]


if __name__ == "__main__":
    query = "Comment remplir mon CRA?"
    history = ""
    print(ask_question(query, history))
