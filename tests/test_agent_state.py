from agents.state import AgentState


def test_agent_state_initial_values():

    state = AgentState(
        current_message="hello"
    )

    assert state.current_message == "hello"
    assert state.round_number == 0
    assert state.previous_response_id is None
    assert state.tool_calls == []


def test_agent_state_starts_tool_round():

    state = AgentState(
        current_message="hello"
    )

    tool_calls = ["call_1"]

    state.start_tool_round(
        tool_calls
    )

    assert state.round_number == 1
    assert state.tool_calls == tool_calls


def test_agent_state_updates_response_id():

    state = AgentState(
        current_message="hello"
    )

    state.update_response_id(
        "response_123"
    )

    assert (
        state.previous_response_id
        == "response_123"
    )