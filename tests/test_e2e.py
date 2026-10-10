"""
End-to-end: HTTP API -> agent -> knowledge base tool -> pgvector -> answer.

Only the two external providers are replaced (scripted LLM and a
deterministic embedder); routing, auth, permissions, indexing, the
production retriever and the database are real.
"""

import json

import pytest

from main import app
from rag.database import RAGSessionLocal
from rag.indexer import DocumentIndexer
from router.ai import get_agent_service
from services.agent_service import (
    AgentService,
    build_knowledge_base_retriever,
)
from services.providers import get_document_indexer
from tests.fakes import FakeEmbedder, ScriptedLLMClient


POLICY = (
    "Astra vacation policy: every employee gets 30 vacation days per "
    "year. Unused vacation days expire at the end of March."
)


@pytest.fixture
def stack(rag_db_session):
    embedder = FakeEmbedder()

    app.dependency_overrides[get_document_indexer] = lambda: DocumentIndexer(
        embedder=embedder,
        session_factory=RAGSessionLocal,
    )

    state = {}

    def use_llm(script):
        llm = ScriptedLLMClient(script)

        service = AgentService(
            llm_client=llm,
            retriever_factory=build_knowledge_base_retriever(
                embedder=embedder,
                session_factory=RAGSessionLocal,
            ),
        )

        app.dependency_overrides[get_agent_service] = lambda: service
        state["llm"] = llm

        return llm

    state["use_llm"] = use_llm

    yield state

    app.dependency_overrides.pop(get_document_indexer, None)
    app.dependency_overrides.pop(get_agent_service, None)


def login(client, username):
    assert client.post(
        "/users/register",
        json={"username": username, "password": "secret123"},
    ).status_code == 200

    token = client.post(
        "/users/login",
        data={"username": username, "password": "secret123"},
    ).json()["access_token"]

    return {"Authorization": f"Bearer {token}"}


def search_script():
    return [
        [
            (
                "search_knowledge_base",
                json.dumps({"query": "how many vacation days per year"}),
            )
        ],
        "Employees get 30 vacation days per year [Source 1].",
    ]


def tool_output(llm):
    return json.loads(llm.calls[1]["message"][0]["output"])


def test_private_document_answer_flow(client, stack):
    owner = login(client, "e2e_owner")

    document_id = client.post(
        "/documents/",
        json={
            "title": "HR policy",
            "category": "HR",
            "body": POLICY,
            "is_private": True,
        },
        headers=owner,
    ).json()["id"]

    indexed = client.post(
        f"/documents/{document_id}/index",
        headers=owner,
    )

    assert indexed.status_code == 200
    assert indexed.json()["chunks"] >= 1

    llm = stack["use_llm"](search_script())

    response = client.post(
        "/ai/agent",
        json={"message": "How many vacation days do I get?"},
        headers=owner,
    )

    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "completed"
    assert body["route"] == "rag"
    assert body["answer"].startswith("Employees get 30 vacation days")
    assert body["sources"][0]["document_id"] == document_id

    # The model really received the indexed text from pgvector.
    output = tool_output(llm)

    assert output["success"] is True
    assert "30 vacation days" in output["data"]["context"]


def test_other_user_cannot_retrieve_private_document(client, stack):
    owner = login(client, "e2e_owner2")
    other = login(client, "e2e_other")

    document_id = client.post(
        "/documents/",
        json={
            "title": "HR policy",
            "category": "HR",
            "body": POLICY,
            "is_private": True,
        },
        headers=owner,
    ).json()["id"]

    client.post(f"/documents/{document_id}/index", headers=owner)

    llm = stack["use_llm"](search_script())

    response = client.post(
        "/ai/agent",
        json={"message": "How many vacation days do I get?"},
        headers=other,
    )

    assert response.status_code == 200
    assert response.json()["sources"] == []

    output = tool_output(llm)

    assert output["data"]["found"] is False
    assert "vacation" not in output["data"]["context"]


def test_deleted_document_is_no_longer_retrieved(client, stack):
    owner = login(client, "e2e_owner3")

    document_id = client.post(
        "/documents/",
        json={"title": "HR policy", "category": "HR", "body": POLICY},
        headers=owner,
    ).json()["id"]

    client.post(f"/documents/{document_id}/index", headers=owner)
    client.delete(f"/documents/{document_id}", headers=owner)

    llm = stack["use_llm"](search_script())

    client.post(
        "/ai/agent",
        json={"message": "How many vacation days do I get?"},
        headers=owner,
    )

    assert tool_output(llm)["data"]["found"] is False
