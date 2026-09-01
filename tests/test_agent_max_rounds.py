from types import SimpleNamespace

import pytest

from agents.agent import Agent
from exceptions import LLMError


class InfiniteToolLLM:

    counter = 0

    def generate_with_tools(
        self,
        message,
        tools,
        instructions=None,
        previous_response_id=None,
    ):

        self.counter += 1

        return SimpleNamespace(
            id=f"response_{self.counter}",
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="get_system_status",
                    arguments="{}",
                    call_id=f"call_{self.counter}",
                )
            ],
            output_text="",
        )


def fake_executor(
    tool_name,
    arguments,
):

    return {
        "success": True,
        "tool_name": tool_name,
        "data": {
            "status": "healthy",
        },
        "error": None,
    }


def test_agent_max_tool_rounds():

    agent = Agent(
        llm_client=InfiniteToolLLM(),
        tool_executor=fake_executor,
    )

    with pytest.raises(LLMError):

        agent.run(
            message="test",
            tools=[
                {
                    "type": "function",
                    "name": "get_system_status",
                    "description": "Get status.",
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