import json

from exceptions import LLMError
from tools.registry import TOOL_REGISTRY


def execute_tool(
    tool_name: str,
    arguments: str,
) -> dict:

    tool = TOOL_REGISTRY.get(tool_name)

    if tool is None:
        raise LLMError(
            f"Unknown tool: {tool_name}"
        )

    try:
        parsed_arguments = json.loads(arguments)

    except json.JSONDecodeError as exc:
        raise LLMError(
            "Invalid tool arguments"
        ) from exc

    if not isinstance(parsed_arguments, dict):
        raise LLMError(
            "Tool arguments must be an object"
        )

    try:
        result = tool(
            **parsed_arguments
        )

        return result

    except TypeError as exc:
        raise LLMError(
            "Invalid tool arguments"
        ) from exc

    except Exception as exc:
        raise LLMError(
            "Tool execution failed"
        ) from exc