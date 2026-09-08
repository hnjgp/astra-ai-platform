from typing import Any

import pytest

from tools.base import BaseTool
from tools.executor import ToolExecutor
from tools.retry import RetryPolicy


class FlakyTool(BaseTool):

    def __init__(
        self,
        failures_before_success: int,
        exception: Exception | None = None,
    ):
        self.calls = 0
        self.failures_before_success = failures_before_success
        self.exception = exception or TimeoutError(
            "temporary timeout"
        )

    @property
    def name(self) -> str:
        return "flaky_tool"

    @property
    def description(self) -> str:
        return "A test tool with temporary failures."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def execute(self, **kwargs: Any) -> dict[str, str]:

        self.calls += 1

        if self.calls <= self.failures_before_success:
            raise self.exception

        return {
            "status": "success",
        }


def test_retry_succeeds_after_temporary_failure(
    monkeypatch,
):

    tool = FlakyTool(
        failures_before_success=1,
    )

    monkeypatch.setattr(
        "tools.executor.get_tool",
        lambda tool_name: tool,
    )

    executor = ToolExecutor(
        retry_policy=RetryPolicy(
            max_retries=2,
        )
    )

    result = executor.execute(
        tool_name="flaky_tool",
        arguments={},
    )

    assert result.success is True
    assert result.tool_name == "flaky_tool"
    assert result.data == {
        "status": "success",
    }

    assert tool.calls == 2


def test_retry_can_handle_multiple_temporary_failures(
    monkeypatch,
):

    tool = FlakyTool(
        failures_before_success=2,
    )

    monkeypatch.setattr(
        "tools.executor.get_tool",
        lambda tool_name: tool,
    )

    executor = ToolExecutor(
        retry_policy=RetryPolicy(
            max_retries=2,
        )
    )

    result = executor.execute(
        tool_name="flaky_tool",
        arguments={},
    )

    assert result.success is True
    assert result.data == {
        "status": "success",
    }

    assert tool.calls == 3


def test_retry_stops_after_max_retries(
    monkeypatch,
):

    tool = FlakyTool(
        failures_before_success=10,
    )

    monkeypatch.setattr(
        "tools.executor.get_tool",
        lambda tool_name: tool,
    )

    executor = ToolExecutor(
        retry_policy=RetryPolicy(
            max_retries=2,
        )
    )

    result = executor.execute(
        tool_name="flaky_tool",
        arguments={},
    )

    assert result.success is False
    assert result.tool_name == "flaky_tool"

    assert result.error is not None
    assert result.error.type == "ToolExecutionError"
    assert result.error.message == "temporary timeout"

    assert tool.calls == 3


def test_non_retryable_error_is_not_retried(
    monkeypatch,
):

    tool = FlakyTool(
        failures_before_success=10,
        exception=ValueError(
            "invalid operation"
        ),
    )

    monkeypatch.setattr(
        "tools.executor.get_tool",
        lambda tool_name: tool,
    )

    executor = ToolExecutor(
        retry_policy=RetryPolicy(
            max_retries=2,
        )
    )

    result = executor.execute(
        tool_name="flaky_tool",
        arguments={},
    )

    assert result.success is False
    assert result.tool_name == "flaky_tool"

    assert result.error is not None
    assert result.error.type == "ToolExecutionError"
    assert result.error.message == "invalid operation"

    assert tool.calls == 1


def test_retry_policy_respects_max_retries():

    policy = RetryPolicy(
        max_retries=2,
    )

    exception = TimeoutError(
        "temporary timeout"
    )

    assert policy.should_retry(
        exception,
        retries_used=0,
    ) is True

    assert policy.should_retry(
        exception,
        retries_used=1,
    ) is True

    assert policy.should_retry(
        exception,
        retries_used=2,
    ) is False


def test_retry_policy_rejects_non_retryable_error():

    policy = RetryPolicy(
        max_retries=2,
    )

    exception = ValueError(
        "invalid operation"
    )

    assert policy.should_retry(
        exception,
        retries_used=0,
    ) is False


def test_retry_policy_rejects_negative_max_retries():

    with pytest.raises(ValueError):

        RetryPolicy(
            max_retries=-1,
        )