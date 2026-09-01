import json
from typing import Any, Callable

from exceptions import LLMError
from schemas import ToolResult


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

        response = self.llm_client.generate_with_tools(
            message=message,
            tools=tools,
            instructions=instructions,
        )

        for round_number in range(
            1,
            max_tool_rounds + 1,
        ):

            tool_calls = [
                item
                for item in response.output
                if item.type == "function_call"
            ]

            if not tool_calls:
                return response.output_text

            print(
                f"TOOL ROUND: {round_number}"
            )

            tool_outputs = []

            for tool_call in tool_calls:

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

            response = self.llm_client.generate_with_tools(
                message=tool_outputs,
                tools=tools,
                instructions=instructions,
                previous_response_id=response.id,
            )

        raise LLMError(
            "Maximum tool execution rounds exceeded"
        )