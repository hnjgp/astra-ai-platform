from tools.executor import ToolExecutor


executor = ToolExecutor()

result = executor.execute(
    tool_name="get_system_status",
    arguments={},
)

assert result["service"] == "Astra"
assert result["status"] == "healthy"

print("TEST: Tool Executor PASS")