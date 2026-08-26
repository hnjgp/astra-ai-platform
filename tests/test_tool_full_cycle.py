from llm.client import LLMClient
from services.ai_service import AIService
from tools.registry import get_tool_definitions


llm_client = LLMClient()

ai_service = AIService(
    llm_client=llm_client
)


tools = get_tool_definitions()


answer = ai_service.generate_with_tools(
    message="وضعیت سیستم Astra را بررسی کن.",
    tools=tools,
)


print("FINAL ANSWER:")
print("----------------------------------------")
print(answer)
print("----------------------------------------")


assert answer
assert "Astra" in answer

print("TEST: Full Tool Execution Cycle PASS")