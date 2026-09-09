# tests/test_agent_context.py

from agents.context import AgentContext


def test_agent_context_initial_values():

    context = AgentContext(
        original_message="Check system status",
        tools=[
            {
                "type": "function",
                "name": "get_system_status",
            }
        ],
    )

    assert context.original_message == (
        "Check system status"
    )

    assert len(context.tools) == 1
    assert context.instructions is None
    assert context.tool_outputs == []


def test_agent_context_stores_instructions():

    context = AgentContext(
        original_message="Check system status",
        tools=[],
        instructions="Be concise.",
    )

    assert context.instructions == "Be concise."


def test_agent_context_stores_tool_outputs():

    context = AgentContext(
        original_message="Check system status",
        tools=[],
    )

    outputs = [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"status": "healthy"}',
        }
    ]

    context.set_tool_outputs(
        outputs
    )

    assert context.tool_outputs == outputs


def test_agent_context_can_clear_tool_outputs():

    context = AgentContext(
        original_message="Check system status",
        tools=[],
    )

    context.set_tool_outputs(
        [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": '{"status": "healthy"}',
            }
        ]
    )

    context.clear_tool_outputs()

    assert context.tool_outputs == []


def test_agent_context_builds_next_model_context():

    context = AgentContext(
        original_message="Check system status",
        tools=[],
    )

    outputs = [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"status": "healthy"}',
        }
    ]

    context.set_tool_outputs(
        outputs
    )

    assert (
        context.build_next_model_context()
        == outputs
    )


def test_agent_context_replaces_previous_tool_outputs():

    context = AgentContext(
        original_message="Check system status",
        tools=[],
    )

    first_outputs = [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"status": "healthy"}',
        }
    ]

    second_outputs = [
        {
            "type": "function_call_output",
            "call_id": "call_2",
            "output": '{"version": "1.0"}',
        }
    ]

    context.set_tool_outputs(
        first_outputs
    )

    context.set_tool_outputs(
        second_outputs
    )

    assert (
        context.build_next_model_context()
        == second_outputs
    )


def test_agent_context_lifecycle():

    context = AgentContext(
        original_message="Check system status",
        tools=[],
    )

    assert (
        context.build_next_model_context()
        == []
    )

    outputs = [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"status": "healthy"}',
        }
    ]

    context.set_tool_outputs(
        outputs
    )

    assert (
        context.build_next_model_context()
        == outputs
    )

    context.clear_tool_outputs()

    assert (
        context.build_next_model_context()
        == []
    )


def test_agent_context_only_composes_tool_outputs():

    context = AgentContext(
        original_message="Check system status",
        tools=[
            {
                "type": "function",
                "name": "get_system_status",
            }
        ],
        instructions="Be concise.",
    )

    outputs = [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"status": "healthy"}',
        }
    ]

    context.set_tool_outputs(
        outputs
    )

    model_context = (
        context.build_next_model_context()
    )

    assert model_context == outputs

    assert context.original_message not in (
        model_context
    )

    assert context.tools not in (
        model_context
    )

    assert context.instructions not in (
        model_context
    )


def test_agent_context_does_not_accumulate_old_outputs():

    context = AgentContext(
        original_message="Check system status",
        tools=[],
    )

    first_outputs = [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"status": "healthy"}',
        }
    ]

    second_outputs = [
        {
            "type": "function_call_output",
            "call_id": "call_2",
            "output": '{"version": "1.0.0"}',
        }
    ]

    context.set_tool_outputs(
        first_outputs
    )

    context.set_tool_outputs(
        second_outputs
    )

    model_context = (
        context.build_next_model_context()
    )

    assert model_context == second_outputs

    assert first_outputs not in (
        model_context
    )