import json

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL

from tools.executor import ToolExecutor


def test_tool_execution_flow():

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    tools = [
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

    response = client.responses.create(
        model=OPENAI_MODEL,
        input="وضعیت سیستم Astra را بررسی کن.",
        tools=tools,
    )

    tool_calls = [
        item
        for item in response.output
        if item.type == "function_call"
    ]

    assert len(tool_calls) == 1

    tool_call = tool_calls[0]

    print("TOOL NAME:", tool_call.name)
    print("ARGUMENTS:", tool_call.arguments)
    print("CALL ID:", tool_call.call_id)

    arguments = json.loads(
        tool_call.arguments
    )

    executor = ToolExecutor()

    result = executor.execute(
        tool_name=tool_call.name,
        arguments=arguments,
    )

    print("TOOL RESULT:", result)

    assert result.success is True
    assert result.tool_name == "get_system_status"
    assert result.data["service"] == "Astra"
    assert result.data["status"] == "healthy"
    assert result.error is None

    print("TEST: Tool Execution Flow PASS")