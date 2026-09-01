import ast
from types import SimpleNamespace

from agents.agent import Agent
from schemas import ToolResult


class FakeLLMClient:

    def __init__(self):

        self.calls = []

    def generate_with_tools(
        self,
        message,
        tools,
        instructions=None,
        previous_response_id=None,
    ):

        self.calls.append(
            {
                "message": message,
                "previous_response_id": (
                    previous_response_id
                ),
            }
        )

        if len(self.calls) == 1:

            return SimpleNamespace(
                id="response_1",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="get_system_status",
                        arguments="{}",
                        call_id="call_1",
                    )
                ],
                output_text="",
            )

        return SimpleNamespace(
            id="response_2",
            output=[],
            output_text=(
                "Astra system is healthy."
            ),
        )


def fake_tool_executor(
    tool_name,
    arguments,
):

    assert tool_name == "get_system_status"

    return ToolResult(
        success=True,
        tool_name=tool_name,
        data={
            "service": "Astra",
            "status": "healthy",
        },
        error=None,
    )


def test_agent_tool_execution():

    llm_client = FakeLLMClient()

    agent = Agent(
        llm_client=llm_client,
        tool_executor=fake_tool_executor,
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

    assert result == (
        "Astra system is healthy."
    )

    assert len(llm_client.calls) == 2

    second_call = llm_client.calls[1]

    assert (
        second_call["previous_response_id"]
        == "response_1"
    )

    output = second_call["message"][0]

    assert (
        output["type"]
        == "function_call_output"
    )

    assert output["call_id"] == "call_1"

    parsed_output = ast.literal_eval(
        output["output"]
    )

    assert parsed_output["success"] is True