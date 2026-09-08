from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    current_message: Any

    round_number: int = 0
    previous_response_id: str | None = None

    tool_calls: list[Any] = field(
        default_factory=list
    )

    def start_tool_round(
        self,
        tool_calls: list[Any],
    ) -> None:
        self.round_number += 1
        self.tool_calls = tool_calls

    def update_response_id(
        self,
        response_id: str | None,
    ) -> None:
        self.previous_response_id = response_id