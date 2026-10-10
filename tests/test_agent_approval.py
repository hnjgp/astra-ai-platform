import pytest

from main import app
from router.ai import get_agent_service
from services.agent_service import AgentService
from tests.fakes import ScriptedLLMClient


REMOVE_CALL = [
    (
        "remove_knowledge_document",
        '{"document_id": 42}',
    )
]


@pytest.fixture
def approval_setup():
    state = {}

    def install(script):
        removed = []

        def remove_document(document_id):
            removed.append(document_id)
            return 3

        llm = ScriptedLLMClient(script)

        service = AgentService(
            llm_client=llm,
            remove_document=remove_document,
        )

        app.dependency_overrides[get_agent_service] = (
            lambda: service
        )

        state.update(
            llm=llm,
            service=service,
            removed=removed,
        )

        return state

    yield install

    app.dependency_overrides.pop(get_agent_service, None)


def admin_headers(client, username="hitl_admin"):
    from database import SessionLocal
    from services.user_service import create_admin

    db = SessionLocal()
    create_admin(db, username, "secret123")
    db.close()

    token = client.post(
        "/users/login",
        data={"username": username, "password": "secret123"},
    ).json()["access_token"]

    return {"Authorization": f"Bearer {token}"}


def user_headers(client, username="hitl_user"):
    client.post(
        "/users/register",
        json={"username": username, "password": "secret123"},
    )

    token = client.post(
        "/users/login",
        data={"username": username, "password": "secret123"},
    ).json()["access_token"]

    return {"Authorization": f"Bearer {token}"}


def start_removal(client, headers):
    return client.post(
        "/ai/agent",
        json={"message": "Remove document 42 from the knowledge base"},
        headers=headers,
    )


def test_sensitive_tool_pauses_for_approval(client, approval_setup):
    setup = approval_setup([REMOVE_CALL])

    response = start_removal(client, admin_headers(client))

    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "approval_required"
    assert body["answer"] is None
    assert body["thread_id"].startswith("user-")
    assert body["pending_tool_calls"] == [
        {
            "call_id": "call_1_0",
            "name": "remove_knowledge_document",
            "arguments": '{"document_id": 42}',
        }
    ]

    # Nothing was executed before a human answered.
    assert setup["removed"] == []


def test_approved_tool_runs_and_agent_finishes(client, approval_setup):
    setup = approval_setup(
        [REMOVE_CALL, "Document 42 was removed."]
    )
    headers = admin_headers(client)

    thread_id = start_removal(client, headers).json()["thread_id"]

    response = client.post(
        f"/ai/agent/approvals/{thread_id}",
        json={"approved": True},
        headers=headers,
    )

    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "completed"
    assert body["answer"] == "Document 42 was removed."
    assert body["tools_used"] == ["remove_knowledge_document"]
    assert setup["removed"] == [42]

    tool_output = setup["llm"].calls[1]["message"][0]["output"]

    assert '"removed_chunks": 3' in tool_output


def test_rejected_tool_is_not_executed(client, approval_setup):
    setup = approval_setup(
        [REMOVE_CALL, "Okay, I did not remove it."]
    )
    headers = admin_headers(client)

    thread_id = start_removal(client, headers).json()["thread_id"]

    response = client.post(
        f"/ai/agent/approvals/{thread_id}",
        json={"approved": False},
        headers=headers,
    )

    assert response.json()["status"] == "completed"
    assert response.json()["tools_used"] == []
    assert setup["removed"] == []

    tool_output = setup["llm"].calls[1]["message"][0]["output"]

    assert "Rejected by a human reviewer" in tool_output


def test_thread_cannot_be_answered_twice(client, approval_setup):
    approval_setup([REMOVE_CALL, "Removed."])
    headers = admin_headers(client)

    thread_id = start_removal(client, headers).json()["thread_id"]

    client.post(
        f"/ai/agent/approvals/{thread_id}",
        json={"approved": True},
        headers=headers,
    )

    response = client.post(
        f"/ai/agent/approvals/{thread_id}",
        json={"approved": True},
        headers=headers,
    )

    assert response.status_code == 404


def test_other_user_cannot_answer_approval(client, approval_setup):
    setup = approval_setup([REMOVE_CALL])

    thread_id = start_removal(
        client,
        admin_headers(client),
    ).json()["thread_id"]

    response = client.post(
        f"/ai/agent/approvals/{thread_id}",
        json={"approved": True},
        headers=admin_headers(client, "another_admin"),
    )

    assert response.status_code == 404
    assert setup["removed"] == []


def test_regular_user_never_sees_sensitive_tool(client, approval_setup):
    setup = approval_setup(["Hi"])

    client.post(
        "/ai/agent",
        json={"message": "Hi"},
        headers=user_headers(client),
    )

    tool_names = {
        tool["name"]
        for tool in setup["llm"].calls[0]["tools"]
    }

    assert "remove_knowledge_document" not in tool_names


def test_unknown_thread_returns_404(client, approval_setup):
    approval_setup([])

    response = client.post(
        "/ai/agent/approvals/user-1-doesnotexist",
        json={"approved": True},
        headers=admin_headers(client),
    )

    assert response.status_code == 404


def test_approval_requires_checkpointer():
    from agents.agent import Agent

    with pytest.raises(ValueError, match="checkpointer"):
        Agent(
            llm_client=None,
            tool_executor=lambda **_: None,
            approval_required_tools={"x"},
        )
