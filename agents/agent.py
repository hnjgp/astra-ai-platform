import json
from dataclasses import dataclass, field
from typing import Any, Callable

from langgraph.errors import GraphRecursionError

from agents.agent_graph import build_agent_graph
from agents.context import AgentContext
from agents.guardrail import AgentGuardrail
from agents.memory_manager import MemoryManager
from agents.planner import Planner
from exceptions import LLMError
from rag.service import RAGService
from schemas import Plan, ToolResult
from tools.knowledge_base import KNOWLEDGE_BASE_TOOL_NAME


@dataclass
class AgentRunResult:
    output_text: str
    tools_used: list[str] = field(default_factory=list)
    used_rag_prompt: bool = False
    recovery_count: int = 0
    plan: Plan | None = None

    @property
    def route(self) -> str:
        """
        Which path the run actually took: normal, tool, rag or tool+rag.
        """

        used_rag = (
            self.used_rag_prompt
            or KNOWLEDGE_BASE_TOOL_NAME in self.tools_used
        )

        used_other_tools = any(
            name != KNOWLEDGE_BASE_TOOL_NAME
            for name in self.tools_used
        )

        if used_rag and used_other_tools:
            return "tool+rag"

        if used_rag:
            return "rag"

        if used_other_tools:
            return "tool"

        return "normal"

    @property
    def recovered(self) -> bool:
        return self.recovery_count > 0


class Agent:
    def __init__(
        self,
        llm_client,
        tool_executor: Callable,
        memory_manager: MemoryManager | None = None,
        rag_service: RAGService | None = None,
        checkpointer=None,
        guardrail: AgentGuardrail | None = None,
        planner: Planner | None = None,
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.memory_manager = memory_manager
        self.rag_service = rag_service
        self.checkpointer = checkpointer
        self.guardrail = (
            guardrail or AgentGuardrail()
        )
        self.planner = planner
        # None = derived from max_tool_rounds for each run.
        self.recursion_limit: int | None = None

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
        role: str = "user",
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

            if role == "user":
                result = self.tool_executor(
                    tool_name=tool_call["name"],
                    arguments=tool_call[
                        "arguments"
                    ],
                )
            else:
                result = self.tool_executor(
                    tool_name=tool_call["name"],
                    arguments=tool_call[
                        "arguments"
                    ],
                    role=role,
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
                    "call_id": tool_call["call_id"],
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
            return [user_message]

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

    def _build_plan_instructions(
        self,
        message: str,
        instructions: str | None,
        planning: bool,
        plan: Plan | None = None,
    ) -> str | None:

        if not planning:
            return instructions

        if self.planner is None:
            raise ValueError(
                "planner is required when planning is True"
            )

        if plan is None:
            plan = self.planner.create_plan(
                message
            )

        plan_text = plan.model_dump_json(
            indent=2,
            ensure_ascii=False,
        )

        if instructions:
            return (
                f"{instructions}\n\n"
                f"Execution plan:\n"
                f"{plan_text}"
            )

        return (
            f"Execution plan:\n"
            f"{plan_text}"
        )

    def run(
        self,
        message: str,
        tools: list[dict],
        max_tool_rounds: int = 5,
        instructions: str | None = None,
        memory_key: str | None = None,
        use_rag: bool = False,
        thread_id: str | None = None,
        role: str = "user",
        planning: bool = False,
    ) -> str:

        return self.run_detailed(
            message=message,
            tools=tools,
            max_tool_rounds=max_tool_rounds,
            instructions=instructions,
            memory_key=memory_key,
            use_rag=use_rag,
            thread_id=thread_id,
            role=role,
            planning=planning,
        ).output_text

    def run_detailed(
        self,
        message: str,
        tools: list[dict],
        max_tool_rounds: int = 5,
        instructions: str | None = None,
        memory_key: str | None = None,
        use_rag: bool = False,
        thread_id: str | None = None,
        role: str = "user",
        planning: bool = False,
    ) -> AgentRunResult:

        self.guardrail.validate_input(
            message
        )

        if (
            self.checkpointer is not None
            and thread_id is None
        ):
            raise ValueError(
                "thread_id is required when "
                "checkpointer is enabled"
            )

        plan = None

        if planning:
            if self.planner is None:
                raise ValueError(
                    "planner is required when planning is True"
                )

            plan = self.planner.create_plan(
                message
            )

            if not plan.steps:
                raise ValueError(
                    "planner returned an empty plan"
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

        model_instructions = (
            self._build_plan_instructions(
                message=message,
                instructions=model_instructions,
                planning=planning,
                plan=plan,
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
            role=role,
            checkpointer=self.checkpointer,
            plan=plan,
            planning=planning,
        )

        config = {
            "recursion_limit": (
                self.recursion_limit
                or max(
                    25,
                    max_tool_rounds * 4 + 5,
                )
            ),
        }

        if thread_id is not None:
            config["configurable"] = {
                "thread_id": thread_id,
            }

        initial_state = {
            "initial_message": initial_message,
            "current_message": initial_message,
            "max_tool_rounds": max_tool_rounds,
            "round_number": 0,
            "previous_response_id": None,
            "tool_calls": [],
            "output_text": None,
            "error": None,
            "role": role,
            "plan": plan,
            "planning": planning,
            "current_step_index": 0,
            "step_results": {},
            "tools_used": [],
            "recovery_count": 0,
        }

        try:
            result = graph.invoke(
                initial_state,
                config=config,
            )

        except GraphRecursionError as exc:
            raise LLMError(
                "Agent exceeded its step limit"
            ) from exc

        output_text = result["output_text"]

        if output_text is None:
            raise LLMError(
                "Agent model response is missing"
            )

        self.guardrail.validate_output(
            output_text
        )

        return AgentRunResult(
            output_text=output_text,
            tools_used=list(result["tools_used"]),
            used_rag_prompt=use_rag,
            recovery_count=result["recovery_count"],
            plan=plan,
        )