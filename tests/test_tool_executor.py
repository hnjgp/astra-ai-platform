from tools.executor import ToolExecutor


def test_tool_executor():

    executor = ToolExecutor()

    result = executor.execute(
        tool_name="get_system_status",
        arguments={},
    )

    print("RESULT:", result)

    assert result.success is True
    assert result.tool_name == "get_system_status"
    assert result.data["service"] == "Astra"
    assert result.data["status"] == "healthy"
    assert result.error is None

    print("TEST: Tool Executor PASS")