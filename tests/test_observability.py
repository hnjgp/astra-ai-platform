import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

from exceptions import LLMError
from llm.client import LLMClient
from observability import JSONFormatter, Metrics, metrics, request_id_var, timed
from tools.executor import ToolExecutor


@pytest.fixture(autouse=True)
def clean_metrics():
    metrics.reset()
    yield
    metrics.reset()


class FakeResponses:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return self.response


def fake_openai(response=None, error=None):
    return SimpleNamespace(
        responses=FakeResponses(response=response, error=error)
    )


def test_timed_records_duration_and_status():
    with timed("work", kind="a"):
        pass

    with pytest.raises(RuntimeError):
        with timed("work", kind="a"):
            raise RuntimeError("boom")

    assert metrics.timer_count("work", kind="a") == 2
    assert metrics.counter("works", kind="a", status="ok") == 1
    assert metrics.counter("works", kind="a", status="error") == 1


def test_prometheus_rendering():
    registry = Metrics()

    registry.increment("http_requests", method="GET", status=200)
    registry.observe("http_request", 0.25, method="GET")

    text = registry.render_prometheus()

    assert 'astra_http_requests_total{method="GET",status="200"} 1' in text
    assert 'astra_http_request_seconds_count{method="GET"} 1' in text
    assert 'astra_http_request_seconds_sum{method="GET"} 0.250000' in text


def test_llm_client_records_tokens_cost_and_latency(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "LLM_INPUT_COST_PER_1M_TOKENS", 2.0)
    monkeypatch.setattr(settings, "LLM_OUTPUT_COST_PER_1M_TOKENS", 8.0)

    client = LLMClient(
        client=fake_openai(
            SimpleNamespace(
                output_text="hi",
                model="test-model",
                usage=SimpleNamespace(
                    input_tokens=1_000,
                    output_tokens=500,
                ),
            )
        ),
        model="test-model",
    )

    assert client.generate("hello", "be brief") == "hi"

    assert metrics.counter(
        "llm_tokens",
        type="input",
        operation="generate",
    ) == 1_000
    assert metrics.counter(
        "llm_tokens",
        type="output",
        operation="generate",
    ) == 500
    # 1000 * 2/1M + 500 * 8/1M = 0.006
    assert metrics.counter(
        "llm_cost_usd",
        operation="generate",
    ) == pytest.approx(0.006)
    assert metrics.timer_count("llm_call", operation="generate") == 1


def test_llm_client_maps_provider_errors():
    error = APIConnectionError(
        request=httpx.Request("POST", "https://example.invalid")
    )

    client = LLMClient(client=fake_openai(error=error))

    with pytest.raises(LLMError, match="Could not connect"):
        client.generate("hello", "x")

    assert metrics.counter(
        "llm_calls",
        operation="generate",
        status="error",
    ) == 1


def test_llm_client_sends_tools_and_previous_response():
    fake = fake_openai(
        SimpleNamespace(output_text="", usage=None, output=[])
    )

    LLMClient(client=fake, model="m").generate_with_tools(
        message=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "name": "t"}],
        instructions="rules",
        previous_response_id="resp_1",
    )

    call = fake.responses.calls[0]

    assert call["model"] == "m"
    assert call["instructions"] == "rules"
    assert call["previous_response_id"] == "resp_1"
    assert call["tools"] == [{"type": "function", "name": "t"}]


def test_tool_calls_are_measured():
    ToolExecutor().execute(tool_name="get_system_status")

    assert metrics.timer_count(
        "tool_call",
        tool="get_system_status",
    ) == 1
    assert metrics.counter(
        "tool_result",
        tool="get_system_status",
        success=True,
    ) == 1


def test_requests_get_request_id_and_metrics(client):
    response = client.get(
        "/health",
        headers={"X-Request-ID": "abc-123"},
    )

    assert response.headers["x-request-id"] == "abc-123"
    assert metrics.counter(
        "http_requests",
        method="GET",
        route="/health",
        status=200,
    ) == 1


def test_unsafe_request_id_is_replaced(client):
    response = client.get(
        "/health",
        headers={"X-Request-ID": "bad id\nwith newline"},
    )

    assert response.headers["x-request-id"] != "bad id\nwith newline"
    assert len(response.headers["x-request-id"]) == 32


def test_route_template_is_used_as_label(client):
    client.get("/documents/12345")

    assert metrics.counter(
        "http_requests",
        method="GET",
        route="/documents/{document_id}",
        status=404,
    ) == 1


def test_metrics_endpoint_requires_api_key(client, monkeypatch):
    import dependency

    monkeypatch.setattr(dependency, "INTERNAL_API_KEY", "m" * 40)

    assert client.get("/metrics").status_code == 422
    assert client.get(
        "/metrics",
        headers={"api-key": "wrong"},
    ).status_code == 401

    client.get("/health")

    response = client.get(
        "/metrics",
        headers={"api-key": "m" * 40},
    )

    assert response.status_code == 200
    assert "astra_http_requests_total" in response.text


def test_json_logs_include_request_id_and_redact_secrets():
    from config import settings

    record = logging.LogRecord(
        name="astra.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="leaked %s",
        args=(settings.SECRET_KEY,),
        exc_info=None,
    )
    record.fields = {"token": "sk-abcdefghijklmnopqrstuvwxyz"}

    token = request_id_var.set("req-1")

    try:
        line = JSONFormatter().format(record)
    finally:
        request_id_var.reset(token)

    entry = json.loads(line)

    assert entry["request_id"] == "req-1"
    assert entry["level"] == "INFO"
    assert settings.SECRET_KEY not in line
    assert "sk-abc" not in line
