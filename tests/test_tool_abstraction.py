from tools.system import (
    ServiceVersionTool,
    SystemStatusTool,
)


system_tool = SystemStatusTool()

system_definition = system_tool.to_definition()

assert system_definition["type"] == "function"
assert system_definition["name"] == "get_system_status"
assert system_definition["description"]
assert isinstance(
    system_definition["parameters"],
    dict,
)

version_tool = ServiceVersionTool()

version_definition = version_tool.to_definition()

assert version_definition["type"] == "function"
assert version_definition["name"] == "get_service_version"
assert version_definition["description"]
assert isinstance(
    version_definition["parameters"],
    dict,
)


print("TEST: Tool Abstraction PASS")