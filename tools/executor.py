import json
from typing import Any

from pydantic import ValidationError

from schemas import ToolError, ToolResult
from tools.registry import get_tool


class ToolExecutor:
    """
    Executes registered Astra tools.
    """

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | str | None = None,
    ) -> ToolResult:

        tool = get_tool(tool_name)

        if arguments is None:
            arguments = {}

        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        try:
            result = tool.execute(**arguments)

            return ToolResult(
                success=True,
                tool_name=tool_name,
                data=result,
                error=None,
            )

        except ValidationError:
            raise

        except Exception as exc:
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
) -> ToolResult:

    executor = ToolExecutor()

    return executor.execute(
        tool_name=tool_name,
        arguments=arguments,
    )