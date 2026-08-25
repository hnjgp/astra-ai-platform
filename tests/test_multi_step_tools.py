from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL
from tools.executor import execute_tool


client = OpenAI(
    api_key=OPENAI_API_KEY
)


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
    },
    {
        "type": "function",
        "name": "get_service_version",
        "description": "Get the current version of Astra.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
]


response = client.responses.create(
    model=OPENAI_MODEL,
    input=(
        "ابتدا وضعیت سیستم Astra را بررسی کن. "
        "بعد نسخه سرویس Astra را بررسی کن. "
        "در پایان نتیجه هر دو را به من بگو."
    ),
    tools=TOOLS,
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

        result = execute_tool(
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
        for tool in TOOLS
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