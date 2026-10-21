from fastapi import APIRouter
from app.ml.rag_engine import rag_engine
from app.schemas.recommendations import AssistantChatRequestSchema, AssistantChatResponseSchema
from app.api.v1.endpoints.operations import increment_assistant_count

router = APIRouter()

@router.post("/chat", response_model=AssistantChatResponseSchema)
def assistant_chat(payload: AssistantChatRequestSchema):
    increment_assistant_count()
    result = rag_engine.query(message=payload.message, language=payload.language)
    return result
