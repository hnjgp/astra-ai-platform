from agents.agent import Agent


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


def test_agent_tool_execution():

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