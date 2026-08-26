from tools.base import BaseTool
from tools.system import (
    ServiceVersionTool,
    SystemStatusTool,
    ServiceInfoTool,
)

TOOL_REGISTRY: dict[str, BaseTool] = {
    "get_system_status": SystemStatusTool(),
    "get_service_version": ServiceVersionTool(),
    "get_service_info": ServiceInfoTool(),
}

def get_tool(tool_name: str) -> BaseTool:
    """
    Return a registered tool by name.
    """
    tool = TOOL_REGISTRY.get(tool_name)

    if tool is None:
        raise KeyError(f"Unknown tool: {tool_name}")

    return tool


def get_tool_definitions() -> list[dict]:
    """
    Return all registered tool definitions for the LLM.
    """
    return [
        tool.to_definition()
        for tool in TOOL_REGISTRY.values()
    ]