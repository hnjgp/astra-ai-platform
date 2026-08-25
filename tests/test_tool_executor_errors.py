from exceptions import LLMError
from tools.executor import execute_tool


try:

    execute_tool(
        tool_name="unknown_tool",
        arguments="{}",
    )

    raise AssertionError(
        "Expected LLMError"
    )

except LLMError as exc:

    assert str(exc) == "Unknown tool: unknown_tool"


print("TEST: Unknown Tool PASS")