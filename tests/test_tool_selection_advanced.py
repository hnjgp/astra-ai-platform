from typing import Any

import pytest
from pydantic import ValidationError

from tools.authorization import ToolAuthorization
from tools.base import BaseTool
from tools.executor import ToolExecutor


class EchoTool(BaseTool):

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo the given text."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
            "additionalProperties": False,
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {"text": kwargs["text"]}


class AdminTool(EchoTool):

    @property
    def name(self) -> str:
        return "admin_only"


def make_executor() -> ToolExecutor:
    return ToolExecutor(
        registry={
            "echo": EchoTool(),
            "admin_only": AdminTool(),
        },
        authorization=ToolAuthorization(
            role_permissions={
                "admin": {"echo", "admin_only"},
                "user": {"echo"},
            }
        ),
    )


def tool_names(definitions: list[dict]) -> set[str]:
    return {
        definition["name"]
        for definition in definitions
    }


def test_model_only_sees_tools_allowed_for_role():
    executor = make_executor()

    assert tool_names(
        executor.tool_definitions("user")
    ) == {"echo"}

    assert tool_names(
        executor.tool_definitions("admin")
    ) == {"echo", "admin_only"}


def test_unknown_role_gets_no_tools():
    executor = make_executor()

    assert executor.tool_definitions("guest") == []


def test_custom_registry_tool_is_executed():
    executor = make_executor()

    result = executor.execute(
        tool_name="echo",
        arguments='{"text": "hello"}',
    )

    assert result.success is True
    assert result.data == {"text": "hello"}


def test_tool_outside_registry_is_rejected():
    executor = make_executor()

    with pytest.raises(
        ValueError,
        match="Tool is not allowed",
    ):
        executor.execute(
            tool_name="get_system_status",
        )


def test_user_cannot_run_admin_tool():
    executor = make_executor()

    with pytest.raises(
        ValueError,
        match="Tool is not allowed for role",
    ):
        executor.execute(
            tool_name="admin_only",
            arguments={"text": "x"},
            role="user",
        )


def test_invalid_json_arguments_are_rejected():
    executor = make_executor()

    with pytest.raises(
        ValueError,
        match="valid JSON",
    ):
        executor.execute(
            tool_name="echo",
            arguments="{not json",
        )


def test_non_object_arguments_are_rejected():
    executor = make_executor()

    with pytest.raises(
        ValueError,
        match="JSON object",
    ):
        executor.execute(
            tool_name="echo",
            arguments="[1, 2]",
        )


def test_empty_argument_string_means_no_arguments():
    executor = ToolExecutor()

    result = executor.execute(
        tool_name="get_system_status",
        arguments="",
    )

    assert result.success is True


def test_arguments_are_validated_before_execution():
    executor = ToolExecutor()

    with pytest.raises(ValidationError):
        executor.execute(
            tool_name="get_service_info",
            arguments={"service_name": 123},
        )
