from typing import Any

from tools.base import BaseTool
from tools.executor import ToolExecutor
from tools.registry import TOOL_REGISTRY


class BrokenTool(BaseTool):

    @property
    def name(self) -> str:
        return "broken_tool"

    @property
    def description(self) -> str:
        return "A tool that intentionally fails."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(
            "Database connection failed"
        )


# اضافه کردن موقت ابزار خراب
TOOL_REGISTRY["broken_tool"] = BrokenTool()


executor = ToolExecutor()


result = executor.execute(
    tool_name="broken_tool",
    arguments={},
)


print("RESULT:", result)


assert result["success"] is False
assert result["tool_name"] == "broken_tool"
assert result["error"] == "Database connection failed"


print("TEST: Tool Error Full Cycle PASS")


# پاک کردن ابزار آزمایشی
del TOOL_REGISTRY["broken_tool"]