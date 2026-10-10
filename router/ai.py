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
from services.providers import get_document_indexer, get_embedder
from schemas import (
    AIGenerateRequest,
    AIGenerateResponse,
    AIChatRequest,
    AgentApprovalRequest,
    AgentRequest,
    AgentResponse,
    AgentSource,
    PendingToolCall,
)


router = APIRouter(
    prefix="/ai",
    tags=["AI"],
)


llm_client = LLMClient()
ai_service = AIService(llm_client)

_agent_service: AgentService | None = None
_agent_checkpointer = None


def configure_agent_checkpointer(checkpointer) -> None:
    """
    Called on startup to store paused runs in a persistent checkpointer.
    """

    global _agent_checkpointer, _agent_service

    _agent_checkpointer = checkpointer
    _agent_service = None


def get_agent_service() -> AgentService:
    global _agent_service

    if _agent_service is None:
        _agent_service = AgentService(
            llm_client=llm_client,
            retriever_factory=build_knowledge_base_retriever(
                embedder=get_embedder(),
                session_factory=RAGSessionLocal,
            ),
            remove_document=get_document_indexer().remove,
            checkpointer=_agent_checkpointer,
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


def to_agent_response(
    result,
    conversation_id: str | None = None,
) -> AgentResponse:

    return AgentResponse(
        status=result.status,
        answer=result.answer,
        route=result.route,
        tools_used=result.tools_used,
        recovered=result.recovered,
        sources=[
            AgentSource(**source.model_dump())
            for source in result.sources
        ],
        conversation_id=conversation_id,
        thread_id=result.thread_id,
        pending_tool_calls=[
            PendingToolCall(**tool_call)
            for tool_call in result.pending_tool_calls
        ],
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

    return to_agent_response(
        result,
        conversation_id=request.conversation_id,
    )


@router.post(
    "/agent/approvals/{thread_id}",
    response_model=AgentResponse,
)
def answer_agent_approval(
    thread_id: str,
    request: AgentApprovalRequest,
    user=Depends(get_current_user),
    db=Depends(get_db),
    agent_service: AgentService = Depends(get_agent_service),
):
    """
    Approve or reject the sensitive tool calls of a paused agent run.
    """

    try:
        result = agent_service.resume(
            thread_id=thread_id,
            approved=request.approved,
            user=user,
            db=db,
        )

    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return to_agent_response(result)
