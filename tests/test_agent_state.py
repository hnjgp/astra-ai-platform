from agents.agent import Agent
from agents.state import AgentState


def test_agent_state_initial_values():
    state = AgentState(
        original_message="وضعیت Astra را بررسی کن.",
        current_message="وضعیت Astra را بررسی کن.",
        tools=[],
    )

    assert state.original_message == "وضعیت Astra را بررسی کن."
    assert state.current_message == "وضعیت Astra را بررسی کن."
    assert state.round_number == 0
    assert state.previous_response_id is None
    assert state.tool_calls == []
    assert state.tool_outputs == []


def test_agent_state_starts_tool_round():
    state = AgentState(
        original_message="وضعیت Astra را بررسی کن.",
        current_message="وضعیت Astra را بررسی کن.",
        tools=[],
    )

    tool_call = {
        "name": "get_system_status",
    }

    state.start_tool_round(
        [tool_call]
    )

    assert state.round_number == 1
    assert state.tool_calls == [tool_call]
    assert state.tool_outputs == []


def test_agent_state_stores_tool_outputs():
    state = AgentState(
        original_message="وضعیت Astra را بررسی کن.",
        current_message="وضعیت Astra را بررسی کن.",
        tools=[],
    )

    outputs = [
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": '{"status": "healthy"}',
        }
    ]

    state.set_tool_outputs(outputs)

    assert state.tool_outputs == outputs


def test_agent_state_updates_response_id():
    state = AgentState(
        original_message="سلام",
        current_message="سلام",
        tools=[],
    )

    state.update_response_id(
        "response_123"
    )

    assert state.previous_response_id == "response_123"


class FakeToolCall:
    type = "function_call"
    name = "get_system_status"
    arguments = "{}"
    call_id = "call_123"


class FakeResponse:
    def __init__(
        self,
        output,
        output_text="",
        response_id="response_1",
    ):
        self.output = output
        self.output_text = output_text
        self.id = response_id


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate_with_tools(
        self,
        message,
        tools,
        instructions=None,
        previous_response_id=None,
    ):
        self.calls += 1

        if self.calls == 1:
            return FakeResponse(
                output=[
                    FakeToolCall()
                ],
                response_id="response_1",
            )

        return FakeResponse(
            output=[],
            output_text="وضعیت Astra سالم است.",
            response_id="response_2",
        )


def test_agent_uses_state_across_tool_round():
    llm = FakeLLM()

    executed_tools = []

    def fake_executor(
        tool_name,
        arguments,
    ):
        executed_tools.append(
            tool_name
        )

        return {
            "success": True,
            "tool_name": tool_name,
            "data": {
                "service": "Astra",
                "status": "healthy",
            },
            "error": None,
        }

    agent = Agent(
        llm_client=llm,
        tool_executor=fake_executor,
    )

    result = agent.run(
        message="وضعیت Astra را بررسی کن.",
        tools=[
            {
                "type": "function",
                "name": "get_system_status",
                "description": "Get system status.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            }
        ],
    )

    assert result == "وضعیت Astra سالم است."

    assert executed_tools == [
        "get_system_status"
    ]

    assert llm.calls == 2