from types import SimpleNamespace

from agents.agent import Agent


class FakeLLMClient:

    def generate_with_tools(
        self,
        message,
        tools,
        instructions=None,
        previous_response_id=None,
    ):

        return SimpleNamespace(
            id="response_1",
            output=[],
            output_text="Final answer",
        )


def test_agent_without_tools():

    agent = Agent(
        llm_client=FakeLLMClient(),
        tool_executor=lambda **kwargs: None,
    )

    result = agent.run(
        message="سلام",
        tools=[],
    )

    assert result == "Final answer"