import pytest

from agents.agent import Agent
from exceptions import LLMError


class FakeToolCall:

    type = "function_call"
    name = "get_system_status"
    arguments = "{}"
    call_id = "call_123"


class FakeResponse:

    output = [
        FakeToolCall()
    ]

    output_text = ""

    id = "response_1"


class FakeLLM:

    def generate_with_tools(
        self,
        message,
        tools,
        instructions=None,
        previous_response_id=None,
    ):

        return FakeResponse()


def test_agent_max_tool_rounds():

    agent = Agent(
        llm_client=FakeLLM(),
        tool_executor=lambda **kwargs: {
            "success": True,
            "tool_name": "get_system_status",
            "data": {
                "status": "healthy"
            },
            "error": None,
        },
    )

    with pytest.raises(LLMError):

        agent.run(
            message="وضعیت سیستم را بررسی کن.",
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
            max_tool_rounds=2,
        )