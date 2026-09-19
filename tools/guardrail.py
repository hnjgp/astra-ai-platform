from collections.abc import Collection

from tools.registry import TOOL_REGISTRY


class ToolGuardrail:
    def __init__(
        self,
        allowed_tools: Collection[str] | None = None,
    ):
        self.allowed_tools = (
            set(allowed_tools)
            if allowed_tools is not None
            else set(TOOL_REGISTRY)
        )

    def check(
        self,
        tool_name: str,
    ) -> None:
        if not tool_name.strip():
            raise ValueError(
                "tool_name must not be empty"
            )

        if tool_name not in self.allowed_tools:
            raise ValueError(
                f"Tool is not allowed: {tool_name}"
            )