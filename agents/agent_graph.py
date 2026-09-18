from typing import Any, TypedDict

from agents.context import AgentContext
from exceptions import LLMError


class AgentGraphState(TypedDict):
    initial_message: list[dict]
    current_message: list[dict]
    max_tool_rounds: int
    round_number: int
    previous_response_id: str | None
    tool_calls: list[dict]
    output_text: str | None
    error: str | None


def build_agent_graph(
    agent,
    context: AgentContext,
    initial_message: list[dict],
    max_tool_rounds: int,
    checkpointer=None,
):
    from langgraph.graph import END, START, StateGraph

    def model_node(
        graph_state: AgentGraphState,
    ) -> dict[str, Any]:

        if (
            graph_state["previous_response_id"]
            is None
        ):
            message = graph_state[
                "initial_message"
            ]

            response = (
                agent.llm_client.generate_with_tools(
                    message=message,
                    tools=context.tools,
                    instructions=context.instructions,
                )
            )

        else:
            message = graph_state[
                "current_message"
            ]

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
            "previous_response_id": response.id,
            "tool_calls": tool_calls,
            "output_text": response.output_text,
            "error": None,
        }

    def route_after_model(
        graph_state: AgentGraphState,
    ) -> str:

        if not graph_state["tool_calls"]:
            return "end"

        if (
            graph_state["round_number"]
            >= graph_state["max_tool_rounds"]
        ):
            raise LLMError(
                "Maximum tool execution rounds exceeded"
            )

        return "tools"

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

    graph.add_edge(
        START,
        "model",
    )

    graph.add_conditional_edges(
        "model",
        route_after_model,
        {
            "tools": "tools",
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

    return graph.compile(
        checkpointer=checkpointer,
    )