from tools.base import BaseTool
from tools.registry import TOOL_REGISTRY
from tools.executor import ToolExecutor


class BrokenTool(BaseTool):

    @property
    def name(self) -> str:
        return "broken_tool"

    @property
    def description(self) -> str:
        return "A tool that always fails."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def execute(self, **kwargs):
        raise RuntimeError("Database connection failed")


def test_tool_execution_error():

    TOOL_REGISTRY["broken_tool"] = BrokenTool()

    executor = ToolExecutor()

    result = executor.execute(
        tool_name="broken_tool",
        arguments={},
    )

    assert result.success is False
    assert result.tool_name == "broken_tool"

    assert result.data is None
    assert result.error is not None

    assert result.error.type == "ToolExecutionError"
    assert "Database connection failed" in result.error.message


print("TEST: Tool Execution Error Handling PASS")