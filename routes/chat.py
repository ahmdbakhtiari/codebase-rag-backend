from fastapi import APIRouter
from pydantic import BaseModel
from services import rag_service

router = APIRouter(prefix="/api/chat", tags=["Chat"])


class chatRequest(BaseModel):
    message: str


@router.post("/")
def chat(request: chatRequest):
    user_question = request.message
    user_answer = rag_service.generate_answer(user_question)
    return {"response": {user_answer}}
