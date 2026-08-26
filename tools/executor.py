import json
from typing import Any

from tools.registry import get_tool


class ToolExecutor:
    """
    Executes registered Astra tools.
    """

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | str | None = None,
    ) -> Any:

        tool = get_tool(tool_name)

        if arguments is None:
            arguments = {}

        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        return tool.execute(**arguments)


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any] | str | None = None,
) -> Any:
    executor = ToolExecutor()

    return executor.execute(
        tool_name=tool_name,
        arguments=arguments,
    )