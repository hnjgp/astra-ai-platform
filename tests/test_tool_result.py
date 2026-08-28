from schemas import ToolError, ToolResult


def test_successful_tool_result():

    result = ToolResult(
        success=True,
        tool_name="get_system_status",
        data={
            "service": "Astra",
            "status": "healthy",
        },
        error=None,
    )

    assert result.success is True
    assert result.tool_name == "get_system_status"
    assert result.data["service"] == "Astra"
    assert result.error is None


def test_failed_tool_result():

    result = ToolResult(
        success=False,
        tool_name="broken_tool",
        data=None,
        error=ToolError(
            type="ToolExecutionError",
            message="Database connection failed",
        ),
    )

    assert result.success is False
    assert result.tool_name == "broken_tool"
    assert result.data is None
    assert result.error.type == "ToolExecutionError"
    assert result.error.message == "Database connection failed"


print("TEST: Tool Result Contract PASS")