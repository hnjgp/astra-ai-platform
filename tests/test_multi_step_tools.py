from llm.client import LLMClient

from services.ai_service import AIService

from tools.registry import get_tool_definitions


def test_multi_step_tool_execution():

    llm_client = LLMClient()

    ai_service = AIService(
        llm_client=llm_client
    )

    tools = get_tool_definitions()

    answer = ai_service.generate_with_tools(
        message=(
            "ابتدا وضعیت سیستم Astra را بررسی کن. "
            "بعد نسخه سرویس Astra را بررسی کن. "
            "در پایان نتیجه هر دو را به من بگو."
        ),
        tools=tools,
        max_tool_rounds=5,
    )

    print()
    print("FINAL ANSWER:")
    print("-" * 40)
    print(answer)
    print("-" * 40)

    assert answer

    assert "Astra" in answer

    print(
        "TEST: Multi-Step Tool Execution PASS"
    )