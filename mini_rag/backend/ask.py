from rag_pipeline import answer_question
from graph import graph


def ask_question(query: str):
    result = graph.invoke({
        "question": query
    })
    return result["answer"]


if __name__ == "__main__":
    query = "Comment remplir mon CRA?"
    print(ask_question(query))
