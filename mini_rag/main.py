from fastapi import FastAPI
from schemas import QuestionRequest, AnswerResponse
from graph import graph

app = FastAPI()


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):

    result = graph.invoke({
        "question": request.question
    })

    return AnswerResponse(answer=result["answer"]) 
