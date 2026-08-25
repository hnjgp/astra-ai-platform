from llm.client import LLMClient
from tools.registry import TOOL_REGISTRY


def build_tools() -> list[dict]:

    return [
        {
            "type": "function",
            "name": tool_name,
            "description": "Get the current status of Astra.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }
        for tool_name in TOOL_REGISTRY
    ]


def main():

    llm_client = LLMClient()

    tools = build_tools()

    response = llm_client.generate_with_tools(
        message="وضعیت Astra چطور است؟",
        tools=tools,
    )

    print("RESPONSE:")
    print(response)

    print()
    print("OUTPUT:")

    for item in response.output:
        print(item)

    tool_calls = [
        item
        for item in response.output
        if item.type == "function_call"
    ]

    assert len(tool_calls) == 1

    tool_call = tool_calls[0]

    assert tool_call.name == "get_system_status"

    print()
    print("TOOL NAME:", tool_call.name)
    print("ARGUMENTS:", tool_call.arguments)
    print("CALL ID:", tool_call.call_id)

    print()
    print("TEST: LLM Tool Call PASS")


if __name__ == "__main__":
    main()