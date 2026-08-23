from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from services.ai_service import AIService
from llm.client import LLMClient
from schemas import (
    AIGenerateRequest,
    AIGenerateResponse,
    AIChatRequest,
)


router = APIRouter(
    prefix="/ai",
    tags=["AI"],
)


llm_client = LLMClient()
ai_service = AIService(llm_client)


@router.post(
    "/generate",
    response_model=AIGenerateResponse,
)
def generate(request: AIGenerateRequest):

    answer = ai_service.generate(request.message)

    return AIGenerateResponse(
        answer=answer
    )


@router.post(
    "/chat",
    response_model=AIGenerateResponse,
)
def chat(request: AIChatRequest):

    answer = ai_service.chat(request.messages)

    return AIGenerateResponse(
        answer=answer
    )


@router.post(
    "/chat/stream",
)
def chat_stream(request: AIChatRequest):

    return StreamingResponse(
        ai_service.chat_stream(request.messages),
        media_type="text/plain",
    )
