import pytest

from main import app
from rate_limit import RateLimitMiddleware
from router.ai import get_agent_service
from tests.fakes import ScriptedLLMClient
from services.agent_service import AgentService


@pytest.fixture
def small_ai_limit(monkeypatch):
    monkeypatch.setattr(RateLimitMiddleware, "max_ai_requests", 2)

    service = AgentService(
        llm_client=ScriptedLLMClient(["ok"] * 10)
    )

    app.dependency_overrides[get_agent_service] = lambda: service

    yield

    app.dependency_overrides.pop(get_agent_service, None)


def token_headers(client, username):
    client.post(
        "/users/register",
        json={"username": username, "password": "secret123"},
    )

    token = client.post(
        "/users/login",
        data={"username": username, "password": "secret123"},
    ).json()["access_token"]

    return {"Authorization": f"Bearer {token}"}


def ask(client, headers):
    return client.post(
        "/ai/agent",
        json={"message": "hi"},
        headers=headers,
    )


def test_ai_endpoints_have_a_stricter_limit(client, small_ai_limit):
    headers = token_headers(client, "limited_user")

    assert ask(client, headers).status_code == 200
    assert ask(client, headers).status_code == 200

    response = ask(client, headers)

    assert response.status_code == 429
    assert 1 <= int(response.headers["Retry-After"]) <= 60


def test_ai_limit_is_per_user(client, small_ai_limit):
    first = token_headers(client, "limited_first")
    second = token_headers(client, "limited_second")

    ask(client, first)
    ask(client, first)

    assert ask(client, first).status_code == 429
    assert ask(client, second).status_code == 200


def test_ai_limit_is_shared_across_ai_paths(client, small_ai_limit):
    headers = token_headers(client, "limited_paths")

    ask(client, headers)
    ask(client, headers)

    response = client.post(
        "/ai/generate",
        json={"message": "hi"},
        headers=headers,
    )

    assert response.status_code == 429


def test_stale_keys_are_cleaned_up(monkeypatch):
    middleware = RateLimitMiddleware(app=None)

    monkeypatch.setattr(RateLimitMiddleware, "cleanup_every", 1)

    RateLimitMiddleware.requests["old"] = [0.0]
    RateLimitMiddleware.requests["fresh"] = [1_000.0]

    middleware._cleanup(now=1_010.0)

    assert "old" not in RateLimitMiddleware.requests
    assert "fresh" in RateLimitMiddleware.requests
