from tools.executor import execute_tool


result = execute_tool(
    tool_name="get_system_status",
    arguments="{}",
)


assert result["service"] == "Astra"
assert result["status"] == "healthy"


print("TEST: Tool Executor PASS")