from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL


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


print("RESPONSE:")
print(response)

print()
print("OUTPUT:")
print(response.output)