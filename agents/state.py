from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """
    Runtime state for a single Agent execution.
    """

    original_message: str
    current_message: Any
    tools: list[dict]
    instructions: str | None = None
    max_tool_rounds: int = 5

    round_number: int = 0
    previous_response_id: str | None = None

    tool_calls: list[Any] = field(default_factory=list)
    tool_outputs: list[dict] = field(default_factory=list)

    def start_tool_round(
        self,
        tool_calls: list[Any],
    ) -> None:
        """
        Start a new tool execution round.
        """

        self.round_number += 1
        self.tool_calls = tool_calls
        self.tool_outputs = []

    def set_tool_outputs(
        self,
        tool_outputs: list[dict],
    ) -> None:
        """
        Store the outputs produced by the current tool round.
        """

        self.tool_outputs = tool_outputs

    def update_response_id(
        self,
        response_id: str | None,
    ) -> None:
        """
        Store the latest model response ID.
        """

        self.previous_response_id = response_id