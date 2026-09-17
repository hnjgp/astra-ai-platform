from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.agent import Agent
from exceptions import LLMError


def make_terminal_response(
    response_id: str,
    output_text: str,
):
    return SimpleNamespace(
        id=response_id,
        output=[],
        output_text=output_text,
    )


def make_tool_response(
    response_id: str,
    tool_name: str,
):
    tool_call = SimpleNamespace(
        type="function_call",
        name=tool_name,
        arguments="{}",
        call_id="call_123",
    )

    return SimpleNamespace(
        id=response_id,
        output=[tool_call],
        output_text="",
    )


def test_agent_graph_completes_without_tool():
    llm_client = Mock()

    llm_client.generate_with_tools.return_value = (
        make_terminal_response(
            response_id="response_1",
            output_text="Hello from Astra",
        )
    )

    tool_executor = Mock()

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
    )

    result = agent.run(
        message="Hello",
        tools=[],
    )

    assert result == "Hello from Astra"

    llm_client.generate_with_tools.assert_called_once()

    tool_executor.assert_not_called()


def test_agent_graph_executes_tool_and_returns_final_response():
    llm_client = Mock()

    llm_client.generate_with_tools.side_effect = [
        make_tool_response(
            response_id="response_1",
            tool_name="get_system_status",
        ),
        make_terminal_response(
            response_id="response_2",
            output_text="The system is healthy.",
        ),
    ]

    tool_executor = Mock(
        return_value={
            "service": "Astra",
            "status": "healthy",
        }
    )

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
    )

    result = agent.run(
        message="What is the system status?",
        tools=[
            {
                "type": "function",
                "name": "get_system_status",
                "description": "Get system status",
                "parameters": {},
            }
        ],
    )

    assert result == "The system is healthy."

    assert (
        llm_client.generate_with_tools.call_count
        == 2
    )

    tool_executor.assert_called_once_with(
        tool_name="get_system_status",
        arguments="{}",
    )

    second_call = (
        llm_client
        .generate_with_tools
        .call_args_list[1]
    )

    assert (
        second_call.kwargs[
            "previous_response_id"
        ]
        == "response_1"
    )


def test_agent_graph_stops_at_max_tool_rounds():
    llm_client = Mock()

    llm_client.generate_with_tools.return_value = (
        make_tool_response(
            response_id="response_1",
            tool_name="get_system_status",
        )
    )

    tool_executor = Mock(
        return_value={
            "service": "Astra",
            "status": "healthy",
        }
    )

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
    )

    with pytest.raises(
        LLMError,
        match="Maximum tool execution rounds exceeded",
    ):
        agent.run(
            message="Keep checking the system.",
            tools=[
                {
                    "type": "function",
                    "name": "get_system_status",
                    "description": "Get system status",
                    "parameters": {},
                }
            ],
            max_tool_rounds=2,
        )

    assert (
        llm_client.generate_with_tools.call_count
        == 3
    )

    assert (
        tool_executor.call_count
        == 2
    )


def test_agent_graph_recovers_from_connection_error():
    llm_client = Mock()

    llm_client.generate_with_tools.side_effect = [
        make_tool_response(
            response_id="response_1",
            tool_name="get_system_status",
        ),
        make_terminal_response(
            response_id="response_2",
            output_text=(
                "I could not access the system status."
            ),
        ),
    ]

    tool_executor = Mock(
        side_effect=ConnectionError(
            "Service temporarily unavailable"
        )
    )

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
    )

    result = agent.run(
        message="What is the system status?",
        tools=[
            {
                "type": "function",
                "name": "get_system_status",
                "description": "Get system status",
                "parameters": {},
            }
        ],
    )

    assert (
        result
        == "I could not access the system status."
    )

    assert (
        llm_client.generate_with_tools.call_count
        == 2
    )

    tool_executor.assert_called_once_with(
        tool_name="get_system_status",
        arguments="{}",
    )

    second_call = (
        llm_client
        .generate_with_tools
        .call_args_list[1]
    )

    assert (
        second_call.kwargs[
            "previous_response_id"
        ]
        == "response_1"
    )

    recovery_message = second_call.kwargs[
        "message"
    ]

    assert any(
        item["type"] == "function_call_output"
        and "Service temporarily unavailable"
        in item["output"]
        for item in recovery_message
    )


def test_agent_graph_does_not_recover_from_value_error():
    llm_client = Mock()

    llm_client.generate_with_tools.return_value = (
        make_tool_response(
            response_id="response_1",
            tool_name="get_system_status",
        )
    )

    tool_executor = Mock(
        side_effect=ValueError(
            "Invalid tool arguments"
        )
    )

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
    )

    with pytest.raises(
        ValueError,
        match="Invalid tool arguments",
    ):
        agent.run(
            message="What is the system status?",
            tools=[
                {
                    "type": "function",
                    "name": "get_system_status",
                    "description": "Get system status",
                    "parameters": {},
                }
            ],
        )

    llm_client.generate_with_tools.assert_called_once()

    tool_executor.assert_called_once_with(
        tool_name="get_system_status",
        arguments="{}",
    )