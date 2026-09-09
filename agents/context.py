# agents/context.py

from dataclasses import dataclass, field


@dataclass
class AgentContext:
    """
    Holds the information available to the agent
    during the current execution.

    Not every field stored in the context is sent
    to the model in every step.
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
        """
        Replace the current tool outputs
        with the outputs of the latest tool round.
        """

        self.tool_outputs = list(
            tool_outputs
        )

    def clear_tool_outputs(self) -> None:
        """
        Remove the current tool outputs.
        """

        self.tool_outputs = []

    def build_next_model_context(
        self,
    ) -> list[dict]:
        """
        Compose the information that should be
        sent to the model for the next step.

        The internal AgentContext may contain more
        information than what is actually sent to
        the model.

        At the current stage, only the latest tool
        outputs are required because the original
        message, tools, instructions, and previous
        response are handled separately.
        """

        return list(
            self.tool_outputs
        )