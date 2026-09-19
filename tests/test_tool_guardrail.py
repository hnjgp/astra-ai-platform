import pytest

from tools.guardrail import ToolGuardrail


def test_registered_tool_passes_guardrail():
    guardrail = ToolGuardrail()

    guardrail.check(
        "get_system_status"
    )


def test_unknown_tool_is_rejected():
    guardrail = ToolGuardrail()

    with pytest.raises(
        ValueError,
        match="Tool is not allowed",
    ):
        guardrail.check(
            "delete_database"
        )


def test_empty_tool_name_is_rejected():
    guardrail = ToolGuardrail()

    with pytest.raises(
        ValueError,
        match="tool_name must not be empty",
    ):
        guardrail.check("")