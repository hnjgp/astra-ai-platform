from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from agents.agent import Agent, AgentRunResult, ApprovalRequiredError
from agents.memory_manager import MemoryManager
from agents.planner import Planner
from prompts.ai import ASTRA_SYSTEM_PROMPT
from rag.context import RAGContext, RAGSource
from rag.embeddings.embedder import Embedder
from rag.retrieval.reranker import KeywordReranker
from rag.retrieval.service import RetrievalService
from rag.service import RAGService
from repository.document import get_readable_document_ids
from tools.authorization import ToolAuthorization
from tools.executor import ToolExecutor
from tools.knowledge_base import (
    KNOWLEDGE_BASE_TOOL_NAME,
    REMOVE_DOCUMENT_TOOL_NAME,
    KnowledgeBaseSearchTool,
    RemoveKnowledgeDocumentTool,
)
from tools.registry import TOOL_REGISTRY


AGENT_INSTRUCTIONS = f"""
{ASTRA_SYSTEM_PROMPT}

Tool rules:

- Use search_knowledge_base when the question is about internal
  documents or project knowledge. Do not use it for general questions.
- Use the system tools only for live service information.
- When you answer from the knowledge base, mention the source numbers.
- If the knowledge base has no relevant information, say so.
""".strip()

MEMORY_TURNS = 5

# Which tools each role may run. The model is only shown these tools.
ROLE_TOOL_PERMISSIONS: dict[str, set[str]] = {
    "admin": {
        *TOOL_REGISTRY,
        KNOWLEDGE_BASE_TOOL_NAME,
        REMOVE_DOCUMENT_TOOL_NAME,
    },
    "user": {
        *TOOL_REGISTRY,
        KNOWLEDGE_BASE_TOOL_NAME,
    },
}

# Tools that pause the run until a human approves them.
APPROVAL_REQUIRED_TOOLS = frozenset(
    {REMOVE_DOCUMENT_TOOL_NAME}
)

# Builds a retrieve function limited to the given document ids
# (None = no restriction).
RetrieverFactory = Callable[
    [Collection[int] | None],
    Callable[[str, int], RAGContext],
]


def build_knowledge_base_retriever(
    embedder: Embedder,
    session_factory,
) -> RetrieverFactory:
    """
    Production retriever: one short-lived RAG database session per search.
    """

    def factory(
        document_ids: Collection[int] | None,
    ) -> Callable[[str, int], RAGContext]:

        def retrieve(query: str, top_k: int) -> RAGContext:
            with session_factory() as db:
                service = RAGService(
                    embedder=embedder,
                    retrieval_service=RetrievalService(
                        db,
                        reranker=KeywordReranker(),
                    ),
                )

                return service.retrieve(
                    question=query,
                    top_k=top_k,
                    document_ids=document_ids,
                )

        return retrieve

    return factory


@dataclass
class AgentAnswer:
    status: Literal["completed", "approval_required"]
    answer: str | None = None
    route: str | None = None
    tools_used: list[str] = field(default_factory=list)
    recovered: bool = False
    sources: list[RAGSource] = field(default_factory=list)
    thread_id: str | None = None
    pending_tool_calls: list[dict] = field(default_factory=list)


class AgentService:
    """
    Runs the agent for one authenticated user.

    Per request it builds the tool set the user is allowed to use,
    including a knowledge base tool restricted to readable documents.
    """

    def __init__(
        self,
        llm_client,
        retriever_factory: RetrieverFactory | None = None,
        remove_document: Callable[[int], int] | None = None,
        memory_manager: MemoryManager | None = None,
        checkpointer=None,
        max_tool_rounds: int = 5,
    ):
        self.llm_client = llm_client
        self.retriever_factory = retriever_factory
        self.remove_document = remove_document
        self.memory_manager = (
            memory_manager or MemoryManager()
        )
        # In-process by default; a paused run is lost on restart.
        # agents/checkpointer.py provides a PostgreSQL saver.
        self.checkpointer = (
            checkpointer or InMemorySaver()
        )
        self.max_tool_rounds = max_tool_rounds
        self.planner = Planner(llm_client)

    def _memory_key(
        self,
        user,
        conversation_id: str | None,
    ) -> str | None:

        if conversation_id is None:
            return None

        return f"user-{user.id}:{conversation_id}"

    def _thread_prefix(self, user) -> str:
        return f"user-{user.id}-"

    def _remember(
        self,
        memory_key: str | None,
        message: str,
        answer: str,
    ) -> None:

        if memory_key is None:
            return

        turns = list(
            self.memory_manager.retrieve(memory_key) or []
        )

        turns.append(f"User: {message}\nAssistant: {answer}")

        self.memory_manager.save(
            memory_key,
            turns[-MEMORY_TURNS:],
        )

    def _build_executor(
        self,
        db,
        user,
    ) -> tuple[ToolExecutor, KnowledgeBaseSearchTool | None]:

        registry = dict(TOOL_REGISTRY)
        knowledge_base_tool = None

        if self.retriever_factory is not None:
            document_ids = get_readable_document_ids(db, user)

            knowledge_base_tool = KnowledgeBaseSearchTool(
                retrieve=self.retriever_factory(document_ids)
            )

            registry[knowledge_base_tool.name] = (
                knowledge_base_tool
            )

        if self.remove_document is not None:
            remove_tool = RemoveKnowledgeDocumentTool(
                remove=self.remove_document
            )

            registry[remove_tool.name] = remove_tool

        executor = ToolExecutor(
            registry=registry,
            authorization=ToolAuthorization(
                role_permissions=ROLE_TOOL_PERMISSIONS
            ),
        )

        return executor, knowledge_base_tool

    def _build_agent(
        self,
        executor: ToolExecutor,
    ) -> Agent:

        return Agent(
            llm_client=self.llm_client,
            tool_executor=executor.execute,
            memory_manager=self.memory_manager,
            planner=self.planner,
            checkpointer=self.checkpointer,
            approval_required_tools=APPROVAL_REQUIRED_TOOLS,
        )

    def _finish(
        self,
        run: Callable[[], AgentRunResult],
        thread_id: str,
        knowledge_base_tool: KnowledgeBaseSearchTool | None,
    ) -> AgentAnswer:

        try:
            result = run()

        except ApprovalRequiredError as exc:
            return AgentAnswer(
                status="approval_required",
                thread_id=exc.thread_id,
                pending_tool_calls=exc.tool_calls,
            )

        except Exception:
            self.checkpointer.delete_thread(thread_id)
            raise

        # Finished runs do not need their checkpoints any more.
        self.checkpointer.delete_thread(thread_id)

        self._remember(
            result.memory_key,
            result.original_message,
            result.output_text,
        )

        return AgentAnswer(
            status="completed",
            answer=result.output_text,
            route=result.route,
            tools_used=result.tools_used,
            recovered=result.recovered,
            sources=(
                list(knowledge_base_tool.sources)
                if knowledge_base_tool is not None
                else []
            ),
        )

    def run(
        self,
        message: str,
        user,
        db,
        conversation_id: str | None = None,
        planning: bool = False,
    ) -> AgentAnswer:

        executor, knowledge_base_tool = self._build_executor(
            db,
            user,
        )

        agent = self._build_agent(executor)

        thread_id = (
            f"{self._thread_prefix(user)}{uuid4().hex}"
        )

        return self._finish(
            run=lambda: agent.run_detailed(
                message=message,
                tools=executor.tool_definitions(user.role),
                max_tool_rounds=self.max_tool_rounds,
                instructions=AGENT_INSTRUCTIONS,
                memory_key=self._memory_key(
                    user,
                    conversation_id,
                ),
                thread_id=thread_id,
                role=user.role,
                planning=planning,
            ),
            thread_id=thread_id,
            knowledge_base_tool=knowledge_base_tool,
        )

    def resume(
        self,
        thread_id: str,
        approved: bool,
        user,
        db,
    ) -> AgentAnswer:

        # Users can only answer approvals for their own runs.
        if not thread_id.startswith(
            self._thread_prefix(user)
        ):
            raise LookupError(
                "no pending approval for this thread"
            )

        executor, knowledge_base_tool = self._build_executor(
            db,
            user,
        )

        agent = self._build_agent(executor)

        return self._finish(
            run=lambda: agent.resume(
                thread_id=thread_id,
                approved=approved,
                role=user.role,
            ),
            thread_id=thread_id,
            knowledge_base_tool=knowledge_base_tool,
        )
