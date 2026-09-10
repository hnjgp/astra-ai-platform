# agents/agent.py

import json
from typing import Any, Callable

from exceptions import LLMError
from schemas import ToolResult

from agents.context import AgentContext
from agents.memory_manager import MemoryManager
from agents.state import AgentState


class Agent:
    def __init__(
        self,
        llm_client,
        tool_executor: Callable,
        memory_manager: MemoryManager | None = None,
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.memory_manager = memory_manager

    def _get_tool_calls(
        self,
        response,
    ) -> list[Any]:

        return [
            item
            for item in response.output
            if item.type == "function_call"
        ]

    def _execute_tool_round(
        self,
        state: AgentState,
        context: AgentContext,
    ) -> list[dict]:

        state.start_tool_round(
            state.tool_calls
        )

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

        context.set_tool_outputs(
            tool_outputs
        )

        return context.build_next_model_context()

    def _is_terminal_response(
        self,
        response,
    ) -> bool:

        return not self._get_tool_calls(response)

    def _build_initial_message(
        self,
        message: str,
        memory_key: str | None,
    ) -> list[dict]:

        user_message = {
            "role": "user",
            "content": message,
        }

        if (
            self.memory_manager is None
            or memory_key is None
        ):
            return [
                user_message
            ]

        memory_context = (
            self.memory_manager.build_context(
                memory_key
            )
        )

        return [
            *memory_context,
            user_message,
        ]

    def run(
        self,
        message: str,
        tools: list[dict],
        max_tool_rounds: int = 5,
        instructions: str | None = None,
        memory_key: str | None = None,
    ) -> str:

        context = AgentContext(
            original_message=message,
            tools=tools,
            instructions=instructions,
        )

        state = AgentState(
            current_message=message,
        )

        initial_message = (
            self._build_initial_message(
                message=message,
                memory_key=memory_key,
            )
        )

        response = self.llm_client.generate_with_tools(
            message=initial_message,
            tools=context.tools,
            instructions=context.instructions,
        )

        state.update_response_id(
            response.id
        )

        for _ in range(
            max_tool_rounds
        ):

            if self._is_terminal_response(
                response
            ):
                return response.output_text

            state.tool_calls = (
                self._get_tool_calls(response)
            )

            self._execute_tool_round(
                state=state,
                context=context,
            )

            state.current_message = (
                context.build_next_model_context()
            )

            response = (
                self.llm_client.generate_with_tools(
                    message=state.current_message,
                    tools=context.tools,
                    instructions=context.instructions,
                    previous_response_id=(
                        state.previous_response_id
                    ),
                )
            )

            state.update_response_id(
                response.id
            )

        raise LLMError(
            "Maximum tool execution rounds exceeded"
        )