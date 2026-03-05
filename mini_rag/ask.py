from rag_pipeline import answer_question

query = "Comment remplir mon CRA?"

answer = answer_question(query)

print(answer)
