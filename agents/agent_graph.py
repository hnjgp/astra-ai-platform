from typing import Any, TypedDict

from agents.context import AgentContext
from agents.state import AgentState
from exceptions import LLMError


class AgentGraphState(TypedDict):
    initial_message: list[dict]
    response: Any | None
    max_tool_rounds: int


def build_agent_graph(
    agent,
    context: AgentContext,
    state: AgentState,
    initial_message: list[dict],
    max_tool_rounds: int,
):
    from langgraph.graph import END, START, StateGraph

    def model_node(
        graph_state: AgentGraphState,
    ) -> dict:
        if state.previous_response_id is None:
            message = graph_state["initial_message"]

            response = (
                agent.llm_client.generate_with_tools(
                    message=message,
                    tools=context.tools,
                    instructions=context.instructions,
                )
            )
        else:
            message = state.current_message

            response = (
                agent.llm_client.generate_with_tools(
                    message=message,
                    tools=context.tools,
                    instructions=context.instructions,
                    previous_response_id=(
                        state.previous_response_id
                    ),
                )
            )

        state.update_response_id(response.id)

        state.tool_calls = (
            agent._get_tool_calls(response)
        )

        return {
            "response": response,
        }

    def route_after_model(
        graph_state: AgentGraphState,
    ) -> str:
        response = graph_state["response"]

        if response is None:
            raise LLMError(
                "Agent model response is missing"
            )

        if agent._is_terminal_response(response):
            return "end"

        if state.round_number >= max_tool_rounds:
            raise LLMError(
                "Maximum tool execution rounds exceeded"
            )

        return "tools"

    def tools_node(
        graph_state: AgentGraphState,
    ) -> dict:
        agent._execute_tool_round(
            state=state,
            context=context,
        )

        state.current_message = (
            context.build_next_model_context()
        )

        return {}

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

    graph.add_edge(
        "tools",
        "model",
    )

    return graph.compile()