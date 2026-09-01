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

        return "Agent response"


class FakeLLM:

    pass


def test_ai_service_uses_agent():

    service = AIService(
        llm_client=FakeLLM()
    )

    fake_agent = FakeAgent()

    service.agent = fake_agent

    result = service.generate_with_tools(
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
    )

    assert result == "Agent response"
    assert fake_agent.called is True