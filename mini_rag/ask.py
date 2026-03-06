from rag_pipeline import answer_question
from graph import graph


query = "Comment remplir mon CRA?"

result = graph.invoke({
    "question": query
})

print(result["answer"])
