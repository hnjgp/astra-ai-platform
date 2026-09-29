from typing import Any, TypedDict

from agents.context import AgentContext
from exceptions import LLMError
from schemas import Plan


class AgentGraphState(TypedDict):
    initial_message: list[dict]
    current_message: list[dict]
    max_tool_rounds: int
    round_number: int
    previous_response_id: str | None
    tool_calls: list[dict]
    output_text: str | None
    error: str | None
    role: str
    plan: Plan | None
    planning: bool
    current_step_index: int
    step_results: dict[int, str]


def build_agent_graph(
    agent,
    context: AgentContext,
    initial_message: list[dict],
    max_tool_rounds: int,
    role: str = "user",
    checkpointer=None,
    plan: Plan | None = None,
    planning: bool = False,
):
    from langgraph.graph import END, START, StateGraph

    def build_step_message(
        step_index: int,
        step_results: dict[int, str],
    ) -> list[dict]:

        if plan is None:
            raise LLMError(
                "Plan is missing for planning execution"
            )

        step = plan.steps[step_index]

        lines = [
            "Execute the current step of the plan.",
            "",
            f"Current step: {step.step}",
            f"Task: {step.description}",
        ]

        if step_results:
            lines.extend(
                [
                    "",
                    "Results from previous steps:",
                ]
            )

            for step_number, result in (
                step_results.items()
            ):
                lines.append(
                    f"Step {step_number} result: "
                    f"{result}"
                )

        lines.extend(
            [
                "",
                "Complete this step using the available "
                "tools when needed.",
                "Use the results from previous steps "
                "when they are relevant.",
            ]
        )

        return [
            {
                "role": "user",
                "content": "\n".join(lines),
            }
        ]

    def model_node(
        graph_state: AgentGraphState,
    ) -> dict[str, Any]:

        if (
            planning
            and plan is not None
            and graph_state[
                "current_step_index"
            ] < len(plan.steps)
        ):
            step_message = build_step_message(
                step_index=graph_state[
                    "current_step_index"
                ],
                step_results=graph_state[
                    "step_results"
                ],
            )

            if (
                graph_state[
                    "previous_response_id"
                ]
                is None
            ):
                message = [
                    *graph_state[
                        "initial_message"
                    ],
                    *step_message,
                ]
            else:
                message = step_message

        elif (
            graph_state["previous_response_id"]
            is None
        ):
            message = graph_state[
                "initial_message"
            ]

        else:
            message = graph_state[
                "current_message"
            ]

        if (
            graph_state["previous_response_id"]
            is None
        ):
            response = (
                agent.llm_client.generate_with_tools(
                    message=message,
                    tools=context.tools,
                    instructions=context.instructions,
                )
            )

        else:
            response = (
                agent.llm_client.generate_with_tools(
                    message=message,
                    tools=context.tools,
                    instructions=context.instructions,
                    previous_response_id=(
                        graph_state[
                            "previous_response_id"
                        ]
                    ),
                )
            )

        tool_calls = (
            agent._serialize_tool_calls(
                response
            )
        )

        return {
            "current_message": message,
            "previous_response_id": response.id,
            "tool_calls": tool_calls,
            "output_text": response.output_text,
            "error": None,
        }

    def route_after_model(
        graph_state: AgentGraphState,
    ) -> str:

        if graph_state["tool_calls"]:

            if (
                graph_state["round_number"]
                >= graph_state["max_tool_rounds"]
            ):
                raise LLMError(
                    "Maximum tool execution rounds exceeded"
                )

            return "tools"

        if graph_state["planning"]:
            return "advance_step"

        return "end"

    def tools_node(
        graph_state: AgentGraphState,
    ) -> dict[str, Any]:

        try:
            next_message = (
                agent._execute_tool_round(
                    tool_calls=graph_state[
                        "tool_calls"
                    ],
                    context=context,
                    role=graph_state["role"],
                )
            )

            return {
                "current_message": next_message,
                "round_number": (
                    graph_state["round_number"]
                    + 1
                ),
                "error": None,
            }

        except (
            TimeoutError,
            ConnectionError,
        ) as error:
            return {
                "error": str(error),
            }

    def route_after_tools(
        graph_state: AgentGraphState,
    ) -> str:

        if graph_state["error"] is not None:
            return "recovery"

        return "model"

    def recovery_node(
        graph_state: AgentGraphState,
    ) -> dict[str, Any]:

        error_message = graph_state["error"]

        if error_message is None:
            raise LLMError(
                "Recovery error is missing"
            )

        tool_outputs = []

        for tool_call in graph_state[
            "tool_calls"
        ]:
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call[
                        "call_id"
                    ],
                    "output": (
                        "Tool execution failed. "
                        f"Error: {error_message}"
                    ),
                }
            )

        context.set_tool_outputs(
            tool_outputs
        )

        return {
            "current_message": (
                context.build_next_model_context()
            ),
            "error": None,
        }

    def advance_step_node(
        graph_state: AgentGraphState,
    ) -> dict[str, Any]:

        if plan is None:
            raise LLMError(
                "Plan is missing for planning execution"
            )

        current_step_index = (
            graph_state["current_step_index"]
        )

        current_step = plan.steps[
            current_step_index
        ]

        step_results = dict(
            graph_state["step_results"]
        )

        output_text = graph_state[
            "output_text"
        ]

        if output_text is not None:
            step_results[
                current_step.step
            ] = output_text

        next_step_index = (
            current_step_index + 1
        )

        if next_step_index >= len(
            plan.steps
        ):
            return {
                "current_step_index": (
                    next_step_index
                ),
                "step_results": step_results,
            }

        next_message = build_step_message(
            step_index=next_step_index,
            step_results=step_results,
        )

        return {
            "current_step_index": (
                next_step_index
            ),
            "step_results": step_results,
            "current_message": next_message,
            "tool_calls": [],
        }

    def route_after_step(
        graph_state: AgentGraphState,
    ) -> str:

        if plan is None:
            raise LLMError(
                "Plan is missing for planning execution"
            )

        if (
            graph_state["current_step_index"]
            >= len(plan.steps)
        ):
            return "end"

        return "model"

    graph = StateGraph(
        AgentGraphState
    )

    graph.add_node(
        "model",
        model_node,
    )

    graph.add_node(
        "tools",
        tools_node,
    )

    graph.add_node(
        "recovery",
        recovery_node,
    )

    graph.add_node(
        "advance_step",
        advance_step_node,
    )

    graph.add_edge(
        START,
        "model",
    )

    graph.add_conditional_edges(
        "model",
        route_after_model,
        {
            "tools": "tools",
            "advance_step": "advance_step",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "model": "model",
            "recovery": "recovery",
        },
    )

    graph.add_edge(
        "recovery",
        "model",
    )

    graph.add_conditional_edges(
        "advance_step",
        route_after_step,
        {
            "model": "model",
            "end": END,
        },
    )

    return graph.compile(
        checkpointer=checkpointer,
    )