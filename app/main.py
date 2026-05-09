from typing import List, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.recommender import SHLRecommender


app = FastAPI(title="SHL Assessment Recommender")

recommender = SHLRecommender()


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool


@app.get("/")
def root():
    return {
        "message": "SHL Assessment Recommender API is running",
        "health": "/health",
        "chat": "/chat",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = [m.dict() for m in request.messages]
    return recommender.chat(messages)
