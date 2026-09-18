import json
from typing import Any, Callable

from agents.agent_graph import build_agent_graph
from agents.context import AgentContext
from agents.memory_manager import MemoryManager
from exceptions import LLMError
from rag.service import RAGService
from schemas import ToolResult


class Agent:
    def __init__(
        self,
        llm_client,
        tool_executor: Callable,
        memory_manager: MemoryManager | None = None,
        rag_service: RAGService | None = None,
        checkpointer=None,
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.memory_manager = memory_manager
        self.rag_service = rag_service
        self.checkpointer = checkpointer

    def _get_tool_calls(
        self,
        response,
    ) -> list[Any]:

        return [
            item
            for item in response.output
            if item.type == "function_call"
        ]

    def _serialize_tool_calls(
        self,
        response,
    ) -> list[dict]:

        return [
            {
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
            for tool_call in self._get_tool_calls(
                response
            )
        ]

    def _execute_tool_round(
        self,
        tool_calls: list[dict],
        context: AgentContext,
    ) -> list[dict]:

        tool_outputs = []

        for tool_call in tool_calls:

            print(
                "TOOL CALL:",
                tool_call["name"],
            )

            print(
                "ARGUMENTS:",
                tool_call["arguments"],
            )

            result = self.tool_executor(
                tool_name=tool_call["name"],
                arguments=tool_call[
                    "arguments"
                ],
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
                    "call_id": tool_call[
                        "call_id"
                    ],
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

        return not self._get_tool_calls(
            response
        )

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

    def _build_instructions(
        self,
        instructions: str | None,
        message: str,
        use_rag: bool,
    ) -> str | None:

        rag_prompt = None

        if use_rag:
            if self.rag_service is None:
                raise ValueError(
                    "rag_service is required when "
                    "use_rag is True"
                )

            rag_prompt = (
                self.rag_service.build_prompt(
                    question=message,
                )
            )

        if instructions and rag_prompt:
            return (
                f"{instructions}\n\n"
                f"{rag_prompt}"
            )

        return instructions or rag_prompt

    def run(
        self,
        message: str,
        tools: list[dict],
        max_tool_rounds: int = 5,
        instructions: str | None = None,
        memory_key: str | None = None,
        use_rag: bool = False,
        thread_id: str | None = None,
    ) -> str:

        if (
            self.checkpointer is not None
            and thread_id is None
        ):
            raise ValueError(
                "thread_id is required when "
                "checkpointer is enabled"
            )

        context = AgentContext(
            original_message=message,
            tools=tools,
            instructions=instructions,
        )

        initial_message = (
            self._build_initial_message(
                message=message,
                memory_key=memory_key,
            )
        )

        model_instructions = (
            self._build_instructions(
                instructions=instructions,
                message=message,
                use_rag=use_rag,
            )
        )

        context.instructions = (
            model_instructions
        )

        graph = build_agent_graph(
            agent=self,
            context=context,
            initial_message=initial_message,
            max_tool_rounds=max_tool_rounds,
            checkpointer=self.checkpointer,
        )

        config = {
            "recursion_limit": max(
                25,
                max_tool_rounds * 4 + 5,
            ),
        }

        if thread_id is not None:
            config["configurable"] = {
                "thread_id": thread_id,
            }

        result = graph.invoke(
            {
                "initial_message": initial_message,
                "current_message": initial_message,
                "max_tool_rounds": max_tool_rounds,
                "round_number": 0,
                "previous_response_id": None,
                "tool_calls": [],
                "output_text": None,
                "error": None,
            },
            config=config,
        )

        output_text = result[
            "output_text"
        ]

        if output_text is None:
            raise LLMError(
                "Agent model response is missing"
            )

        return output_text