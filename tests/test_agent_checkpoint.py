from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.agent import Agent
from langgraph.checkpoint.memory import InMemorySaver


def make_terminal_response(
    response_id: str,
    output_text: str,
):
    return SimpleNamespace(
        id=response_id,
        output=[],
        output_text=output_text,
    )


def test_agent_requires_thread_id_when_checkpointer_is_enabled():
    llm_client = Mock()

    tool_executor = Mock()

    checkpointer = InMemorySaver()

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
        checkpointer=checkpointer,
    )

    with pytest.raises(
        ValueError,
        match=(
            "thread_id is required when "
            "checkpointer is enabled"
        ),
    ):
        agent.run(
            message="Hello",
            tools=[],
        )


def test_agent_saves_state_with_thread_id():
    llm_client = Mock()

    llm_client.generate_with_tools.return_value = (
        make_terminal_response(
            response_id="response_1",
            output_text="Hello from Astra",
        )
    )

    tool_executor = Mock()

    checkpointer = InMemorySaver()

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
        checkpointer=checkpointer,
    )

    result = agent.run(
        message="Hello",
        tools=[],
        thread_id="conversation-1",
    )

    assert result == "Hello from Astra"

    config = {
        "configurable": {
            "thread_id": "conversation-1",
        },
    }

    saved = checkpointer.get_tuple(
        config
    )

    assert saved is not None

    channel_values = (
        saved.checkpoint["channel_values"]
    )

    assert (
        channel_values["output_text"]
        == "Hello from Astra"
    )

    assert (
        channel_values["previous_response_id"]
        == "response_1"
    )


def test_agent_keeps_threads_separate():
    llm_client = Mock()

    llm_client.generate_with_tools.side_effect = [
        make_terminal_response(
            response_id="response_1",
            output_text="Response for thread one",
        ),
        make_terminal_response(
            response_id="response_2",
            output_text="Response for thread two",
        ),
    ]

    tool_executor = Mock()

    checkpointer = InMemorySaver()

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
        checkpointer=checkpointer,
    )

    first_result = agent.run(
        message="Hello from one",
        tools=[],
        thread_id="conversation-1",
    )

    second_result = agent.run(
        message="Hello from two",
        tools=[],
        thread_id="conversation-2",
    )

    assert (
        first_result
        == "Response for thread one"
    )

    assert (
        second_result
        == "Response for thread two"
    )

    first_config = {
        "configurable": {
            "thread_id": "conversation-1",
        },
    }

    second_config = {
        "configurable": {
            "thread_id": "conversation-2",
        },
    }

    first_saved = checkpointer.get_tuple(
        first_config
    )

    second_saved = checkpointer.get_tuple(
        second_config
    )

    assert first_saved is not None
    assert second_saved is not None

    first_values = (
        first_saved.checkpoint["channel_values"]
    )

    second_values = (
        second_saved.checkpoint["channel_values"]
    )

    assert (
        first_values["output_text"]
        == "Response for thread one"
    )

    assert (
        second_values["output_text"]
        == "Response for thread two"
    )