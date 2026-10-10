from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from database import get_db
from dependency import get_current_user
from services.ai_service import AIService
from services.agent_service import (
    AgentService,
    build_knowledge_base_retriever,
)
from llm.client import LLMClient
from rag.database import RAGSessionLocal
from rag.embeddings.embedder import OpenAIEmbedder
from schemas import (
    AIGenerateRequest,
    AIGenerateResponse,
    AIChatRequest,
    AgentRequest,
    AgentResponse,
    AgentSource,
)


router = APIRouter(
    prefix="/ai",
    tags=["AI"],
)


llm_client = LLMClient()
ai_service = AIService(llm_client)

_agent_service: AgentService | None = None


def get_agent_service() -> AgentService:
    global _agent_service

    if _agent_service is None:
        _agent_service = AgentService(
            llm_client=llm_client,
            retriever_factory=build_knowledge_base_retriever(
                embedder=OpenAIEmbedder(llm_client.client),
                session_factory=RAGSessionLocal,
            ),
        )

    return _agent_service


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


@router.post(
    "/agent",
    response_model=AgentResponse,
)
def run_agent(
    request: AgentRequest,
    user=Depends(get_current_user),
    db=Depends(get_db),
    agent_service: AgentService = Depends(get_agent_service),
):
    try:
        result = agent_service.run(
            message=request.message,
            user=user,
            db=db,
            conversation_id=request.conversation_id,
            planning=request.planning,
        )

    except ValueError as exc:
        # Guardrail and validation failures are safe to show the caller.
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return AgentResponse(
        answer=result.answer,
        route=result.route,
        tools_used=result.tools_used,
        recovered=result.recovered,
        sources=[
            AgentSource(**source.model_dump())
            for source in result.sources
        ],
        conversation_id=request.conversation_id,
    )
