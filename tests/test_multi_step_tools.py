from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL
from tools.executor import ToolExecutor
from tools.registry import get_tool_definitions


client = OpenAI(
    api_key=OPENAI_API_KEY
)

executor = ToolExecutor()

tools = get_tool_definitions()


response = client.responses.create(
    model=OPENAI_MODEL,
    input=(
        "ابتدا وضعیت سیستم Astra را بررسی کن. "
        "بعد نسخه سرویس Astra را بررسی کن. "
        "در پایان نتیجه هر دو را به من بگو."
    ),
    tools=tools,
    tool_choice="required",
    parallel_tool_calls=False,
)


tool_rounds = 0
used_tools = set()


while True:

    tool_calls = [
        item
        for item in response.output
        if item.type == "function_call"
    ]

    if not tool_calls:
        break

    tool_rounds += 1

    print()
    print(f"TOOL ROUND: {tool_rounds}")

    tool_outputs = []

    for tool_call in tool_calls:

        print(
            "TOOL CALL:",
            tool_call.name,
        )

        print(
            "ARGUMENTS:",
            tool_call.arguments,
        )

        result = executor.execute(
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
        )

        used_tools.add(tool_call.name)

        print(
            "TOOL RESULT:",
            result,
        )

        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": str(result),
            }
        )

    remaining_tools = [
        tool
        for tool in tools
        if tool["name"] not in used_tools
    ]

    if not remaining_tools:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=tool_outputs,
            previous_response_id=response.id,
        )
        break

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=tool_outputs,
        previous_response_id=response.id,
        tools=remaining_tools,
        tool_choice="required",
        parallel_tool_calls=False,
    )


print()
print("FINAL ANSWER:")
print("-" * 40)
print(response.output_text)
print("-" * 40)


assert tool_rounds >= 2
assert "get_system_status" in used_tools
assert "get_service_version" in used_tools


print(
    "TEST: Multi-Step Tool Execution PASS"
)