import json
from typing import Any

from pydantic import ValidationError

from schemas import ToolError, ToolResult
from tools.authorization import ToolAuthorization
from tools.base import BaseTool
from tools.guardrail import ToolGuardrail
from tools.registry import TOOL_REGISTRY, get_tool
from tools.retry import RetryPolicy


class ToolExecutor:
    """
    Executes registered Astra tools.
    """

    def __init__(
        self,
        retry_policy: RetryPolicy | None = None,
        guardrail: ToolGuardrail | None = None,
        authorization: ToolAuthorization | None = None,
        registry: dict[str, BaseTool] | None = None,
    ):
        self.registry = (
            registry
            if registry is not None
            else TOOL_REGISTRY
        )
        self.retry_policy = (
            retry_policy or RetryPolicy()
        )
        self.guardrail = (
            guardrail
            or ToolGuardrail(
                allowed_tools=self.registry
            )
        )
        self.authorization = (
            authorization or ToolAuthorization()
        )

    def tool_definitions(
        self,
        role: str = "user",
    ) -> list[dict]:
        """
        Return only the tools this role may actually run.

        The model never sees tools it is not allowed to call, so it
        cannot select them in the first place.
        """

        allowed = (
            self.guardrail.allowed_tools
            & self.authorization.allowed_tools(role)
        )

        return [
            tool.to_definition()
            for name, tool in self.registry.items()
            if name in allowed
        ]

    def _get_tool(
        self,
        tool_name: str,
    ) -> BaseTool:

        if self.registry is TOOL_REGISTRY:
            return get_tool(tool_name)

        tool = self.registry.get(tool_name)

        if tool is None:
            raise KeyError(
                f"Unknown tool: {tool_name}"
            )

        return tool

    def _parse_arguments(
        self,
        arguments: dict[str, Any] | str | None,
    ) -> dict[str, Any]:

        if arguments is None:
            return {}

        if isinstance(arguments, str):
            if not arguments.strip():
                return {}

            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Tool arguments must be valid JSON"
                ) from exc

        if not isinstance(arguments, dict):
            raise ValueError(
                "Tool arguments must be a JSON object"
            )

        return arguments

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | str | None = None,
        role: str = "user",
    ) -> ToolResult:

        self.guardrail.check(tool_name)

        self.authorization.check(
            role=role,
            tool_name=tool_name,
        )

        tool = self._get_tool(tool_name)

        arguments = self._parse_arguments(
            arguments
        )

        if tool.input_schema is not None:
            tool.input_schema.model_validate(
                arguments
            )

        retries_used = 0

        while True:
            try:
                result = tool.execute(
                    **arguments
                )

                return ToolResult(
                    success=True,
                    tool_name=tool_name,
                    data=result,
                    error=None,
                )

            except ValidationError:
                raise

            except Exception as exc:

                if self.retry_policy.should_retry(
                    exception=exc,
                    retries_used=retries_used,
                ):
                    retries_used += 1
                    continue

                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    data=None,
                    error=ToolError(
                        type="ToolExecutionError",
                        message=str(exc),
                    ),
                )


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any] | str | None = None,
    role: str = "user",
) -> ToolResult:

    executor = ToolExecutor()

    return executor.execute(
        tool_name=tool_name,
        arguments=arguments,
        role=role,
    )
