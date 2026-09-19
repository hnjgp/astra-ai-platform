import json
from typing import Any

from pydantic import ValidationError

from schemas import ToolError, ToolResult
from tools.authorization import ToolAuthorization
from tools.guardrail import ToolGuardrail
from tools.registry import get_tool
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
    ):
        self.retry_policy = (
            retry_policy or RetryPolicy()
        )
        self.guardrail = (
            guardrail or ToolGuardrail()
        )
        self.authorization = (
            authorization or ToolAuthorization()
        )

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

        tool = get_tool(tool_name)

        if arguments is None:
            arguments = {}

        if isinstance(arguments, str):
            arguments = json.loads(arguments)

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