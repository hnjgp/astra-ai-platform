from llm.client import LLMClient

from services.ai_service import AIService

from tools.registry import get_tool_definitions


def test_full_tool_execution_cycle():

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
    print("-" * 40)
    print(answer)
    print("-" * 40)

    assert answer
    assert "Astra" in answer

    print(
        "TEST: Full Tool Execution Cycle PASS"
    )