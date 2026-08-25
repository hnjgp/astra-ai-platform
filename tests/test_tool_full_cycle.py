from llm.client import LLMClient
from services.ai_service import AIService


TOOLS = [
    {
        "type": "function",
        "name": "get_system_status",
        "description": "Get the current status of Astra.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }
]


llm_client = LLMClient()

ai_service = AIService(
    llm_client=llm_client
)


answer = ai_service.generate_with_tools(
    message="وضعیت سیستم Astra را بررسی کن.",
    tools=TOOLS,
)


print("FINAL ANSWER:")
print("----------------------------------------")
print(answer)
print("----------------------------------------")

assert answer
assert "Astra" in answer

print(
    "TEST: Full Tool Execution Cycle PASS"
)