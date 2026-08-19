from fastapi import APIRouter

from schemas import AIGenerateRequest, AIGenerateResponse
from services.ai_service import AIService
from llm.client import LLMClient


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