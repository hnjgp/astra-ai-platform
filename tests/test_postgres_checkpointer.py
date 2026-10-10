import os
from types import SimpleNamespace

from agents.checkpointer import postgres_checkpointer, to_psycopg_url
from services.agent_service import AgentService
from tests.fakes import ScriptedLLMClient


ADMIN = SimpleNamespace(id=1, role="admin")


def test_url_conversion():
    assert to_psycopg_url(
        "postgresql+psycopg://u:p@h:5432/db"
    ) == "postgresql://u:p@h:5432/db"


def test_paused_run_survives_a_restart():
    database_url = os.environ["RAG_DATABASE_URL"]
    removed = []

    def remove_document(document_id):
        removed.append(document_id)
        return 1

    # First "process": the run pauses for approval.
    with postgres_checkpointer(database_url) as saver:
        paused = AgentService(
            llm_client=ScriptedLLMClient(
                [[("remove_knowledge_document", '{"document_id": 4}')]]
            ),
            remove_document=remove_document,
            checkpointer=saver,
        ).run("Remove document 4", ADMIN, db=None)

    assert paused.status == "approval_required"
    assert removed == []

    # Second "process" with a new connection resumes the same thread.
    with postgres_checkpointer(database_url) as saver:
        finished = AgentService(
            llm_client=ScriptedLLMClient(["Document 4 was removed."]),
            remove_document=remove_document,
            checkpointer=saver,
        ).resume(paused.thread_id, True, ADMIN, db=None)

    assert finished.status == "completed"
    assert finished.answer == "Document 4 was removed."
    assert removed == [4]


def test_app_uses_postgres_checkpointer_when_configured(monkeypatch):
    from fastapi.testclient import TestClient
    from langgraph.checkpoint.postgres import PostgresSaver

    import router.ai
    from config import settings
    from main import app

    monkeypatch.setattr(settings, "AGENT_CHECKPOINTER", "postgres")
    monkeypatch.setattr(settings, "RAG_DATABASE_URL", os.environ["RAG_DATABASE_URL"])

    with TestClient(app):
        assert isinstance(
            router.ai.get_agent_service().checkpointer,
            PostgresSaver,
        )

    assert router.ai._agent_checkpointer is None
