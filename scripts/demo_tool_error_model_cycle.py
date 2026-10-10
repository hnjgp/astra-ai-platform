from typing import Any

from llm.client import LLMClient
from services.ai_service import AIService
from tools.base import BaseTool
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


# اضافه کردن ابزار خراب برای تست
TOOL_REGISTRY["broken_tool"] = BrokenTool()


try:

    llm_client = LLMClient()

    ai_service = AIService(
        llm_client=llm_client
    )

    tools = [
        {
            "type": "function",
            "name": "broken_tool",
            "description": (
                "Check the system database connection. "
                "This tool may fail."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }
    ]

    answer = ai_service.generate_with_tools(
        message=(
            "لطفاً وضعیت اتصال پایگاه داده "
            "Astra را بررسی کن و نتیجه را بگو."
        ),
        tools=tools,
    )

    print()
    print("FINAL ANSWER:")
    print("-" * 40)
    print(answer)
    print("-" * 40)

    assert answer
    assert len(answer.strip()) > 0

    print(
        "TEST: Tool Error Model Cycle PASS"
    )

finally:

    # حذف ابزار آزمایشی
    del TOOL_REGISTRY["broken_tool"]