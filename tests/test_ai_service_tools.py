from services.ai_service import AIService


class FakeLLMClient:

    def generate_with_tool_execution(
        self,
        message,
        tools,
        tool_executor,
        max_tool_rounds,
        instructions,
    ):

        assert message == "وضعیت سیستم Astra را بررسی کن."

        assert len(tools) >= 1

        assert callable(tool_executor)

        assert max_tool_rounds == 5

        return "وضعیت Astra سالم است."


client = FakeLLMClient()

service = AIService(
    llm_client=client
)


answer = service.generate_with_tools(
    message="وضعیت سیستم Astra را بررسی کن."
)


print("ANSWER:")
print(answer)

assert answer == "وضعیت Astra سالم است."

print(
    "TEST: AIService Tool Routing PASS"
)