from tools.registry import TOOL_REGISTRY


assert "get_system_status" in TOOL_REGISTRY

tool = TOOL_REGISTRY["get_system_status"]

result = tool()

assert result["service"] == "Astra"
assert result["status"] == "healthy"


print("TEST: Tool Registry PASS")