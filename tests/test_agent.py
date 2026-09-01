from agents.agent import Agent


class FakeLLM:

    def generate_with_tools(
        self,
        message,
        tools,
        instructions=None,
        previous_response_id=None,
    ):

        class Response:

            output = []

            output_text = "پاسخ مستقیم Agent"

            id = "response_1"

        return Response()


def test_agent_without_tools():

    agent = Agent(
        llm_client=FakeLLM(),
        tool_executor=lambda **kwargs: None,
    )

    result = agent.run(
        message="سلام",
        tools=[],
    )

    assert result == "پاسخ مستقیم Agent"