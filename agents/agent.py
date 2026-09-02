import json
from typing import Any, Callable

from exceptions import LLMError
from schemas import ToolResult

from agents.state import AgentState


class Agent:
    def __init__(
        self,
        llm_client,
        tool_executor: Callable,
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor

    def _get_tool_calls(
        self,
        response,
    ) -> list[Any]:
        """
        Extract tool calls from the model response.
        """

        return [
            item
            for item in response.output
            if item.type == "function_call"
        ]

    def _execute_tool_round(
        self,
        state: AgentState,
    ) -> list[dict]:
        """
        Execute all tool calls from the current round.
        """

        state.round_number += 1
        state.tool_outputs = []

        print(
            f"TOOL ROUND: {state.round_number}"
        )

        tool_outputs = []

        for tool_call in state.tool_calls:
            print(
                "TOOL CALL:",
                tool_call.name,
            )

            print(
                "ARGUMENTS:",
                tool_call.arguments,
            )

            result = self.tool_executor(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
            )

            print(
                "TOOL RESULT:",
                result,
            )

            if isinstance(result, ToolResult):
                output = result.model_dump()
            else:
                output = result

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": json.dumps(
                        output,
                        ensure_ascii=False,
                    ),
                }
            )

        state.set_tool_outputs(tool_outputs)

        return state.tool_outputs

    def _is_terminal_response(
        self,
        response,
    ) -> bool:
        """
        Return True when the model produced a final response
        without requesting any tool.
        """

        return not self._get_tool_calls(response)

    def run(
        self,
        message: str,
        tools: list[dict],
        max_tool_rounds: int = 5,
        instructions: str | None = None,
    ) -> str:

        state = AgentState(
            original_message=message,
            current_message=message,
            tools=tools,
            instructions=instructions,
            max_tool_rounds=max_tool_rounds,
        )

        response = self.llm_client.generate_with_tools(
            message=state.current_message,
            tools=state.tools,
            instructions=state.instructions,
        )

        state.update_response_id(response.id)

        for _ in range(
            state.max_tool_rounds
        ):
            if self._is_terminal_response(response):
                return response.output_text

            state.tool_calls = self._get_tool_calls(response)

            self._execute_tool_round(state)

            state.current_message = state.tool_outputs

            response = self.llm_client.generate_with_tools(
                message=state.current_message,
                tools=state.tools,
                instructions=state.instructions,
                previous_response_id=state.previous_response_id,
            )

            state.update_response_id(response.id)

        raise LLMError(
            "Maximum tool execution rounds exceeded"
        )