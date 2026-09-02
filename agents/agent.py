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
            1,
            state.max_tool_rounds + 1,
        ):
            tool_calls = [
                item
                for item in response.output
                if item.type == "function_call"
            ]

            if not tool_calls:
                return response.output_text

            state.start_tool_round(tool_calls)

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