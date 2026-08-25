from tools.system import (
    get_service_version,
    get_system_status,
)


TOOL_REGISTRY = {
    "get_system_status": get_system_status,
    "get_service_version": get_service_version,
}


TOOL_DEFINITIONS = [
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


def get_tool_definitions() -> list[dict]:
    return TOOL_DEFINITIONS