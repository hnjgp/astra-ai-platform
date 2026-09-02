import pytest

from agents.agent import Agent
from exceptions import LLMError


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
            output_text="پاسخ نهایی",
            response_id="response_2",
        )


class InfiniteToolLLM:
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

        return FakeResponse(
            output=[
                FakeToolCall()
            ],
            response_id=f"response_{self.calls}",
        )


def fake_executor(
    tool_name,
    arguments,
):
    return {
        "success": True,
        "tool_name": tool_name,
        "data": {
            "service": "Astra",
            "status": "healthy",
        },
        "error": None,
    }


def test_agent_stops_on_final_response():
    llm = FakeLLM()

    agent = Agent(
        llm_client=llm,
        tool_executor=fake_executor,
    )

    result = agent.run(
        message="وضعیت Astra را بررسی کن.",
        tools=[],
    )

    assert result == "پاسخ نهایی"
    assert llm.calls == 2


def test_agent_stops_after_max_tool_rounds():
    llm = InfiniteToolLLM()

    agent = Agent(
        llm_client=llm,
        tool_executor=fake_executor,
    )

    with pytest.raises(
        LLMError,
        match="Maximum tool execution rounds exceeded",
    ):
        agent.run(
            message="وضعیت Astra را بررسی کن.",
            tools=[],
            max_tool_rounds=2,
        )

    assert llm.calls == 3