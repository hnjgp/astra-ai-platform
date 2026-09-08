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

    context.set_tool_outputs(outputs)

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