from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    """
    Holds the information available to the agent
    during the current execution.
    """

    original_message: str
    tools: list[dict]
    instructions: str | None = None

    tool_outputs: list[dict] = field(
        default_factory=list
    )

    def set_tool_outputs(
        self,
        tool_outputs: list[dict],
    ) -> None:
        self.tool_outputs = tool_outputs

    def clear_tool_outputs(self) -> None:
        self.tool_outputs = []