import pytest
from unittest.mock import Mock

from tools.executor import ToolExecutor
from tools.guardrail import ToolGuardrail


def test_tool_executor():

    executor = ToolExecutor()

    result = executor.execute(
        tool_name="get_system_status",
        arguments={},
    )

    print("RESULT:", result)

    assert result.success is True
    assert result.tool_name == "get_system_status"
    assert result.data["service"] == "Astra"
    assert result.data["status"] == "healthy"
    assert result.error is None

    print("TEST: Tool Executor PASS")


def test_tool_executor_rejects_unknown_tool():

    executor = ToolExecutor()

    with pytest.raises(
        ValueError,
        match="Tool is not allowed",
    ):
        executor.execute(
            tool_name="delete_database",
            arguments={},
        )


def test_guardrail_runs_before_tool_execution():

    guardrail = Mock(
        spec=ToolGuardrail
    )

    guardrail.check.side_effect = ValueError(
        "Tool is not allowed"
    )

    executor = ToolExecutor(
        guardrail=guardrail
    )

    with pytest.raises(
        ValueError,
        match="Tool is not allowed",
    ):
        executor.execute(
            tool_name="get_system_status",
            arguments={},
        )

    guardrail.check.assert_called_once_with(
        "get_system_status"
    )