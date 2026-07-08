import uvicorn
from fastapi import FastAPI
from question_generate.questions import router as question_router

app = FastAPI(
    title="TEST QUESTION",
    version="1.0.0"
)

app.include_router(question_router)

