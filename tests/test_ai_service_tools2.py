from services.ai_service import AIService


class FakeAgent:

    def __init__(self):

        self.called = False

    def run(
        self,
        message,
        tools,
        max_tool_rounds,
        instructions,
    ):

        self.called = True

        assert message == "وضعیت Astra را بررسی کن."

        assert tools

        assert max_tool_rounds == 5

        return "Astra is healthy."


class FakeLLMClient:
    pass


def test_ai_service_uses_agent():

    service = AIService(
        llm_client=FakeLLMClient()
    )

    fake_agent = FakeAgent()

    service.agent = fake_agent

    result = service.generate_with_tools(
        message="وضعیت Astra را بررسی کن."
    )

    assert result == "Astra is healthy."

    assert fake_agent.called is True