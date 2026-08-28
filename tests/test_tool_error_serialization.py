import json

from schemas import ToolError, ToolResult


def test_tool_error_serialization():

    result = ToolResult(
        success=False,
        tool_name="broken_tool",
        data=None,
        error=ToolError(
            type="ToolExecutionError",
            message="Database connection failed",
        ),
    )

    json_result = result.model_dump_json()

    print("ERROR JSON RESULT:")
    print(json_result)

    parsed = json.loads(json_result)

    assert parsed["success"] is False
    assert parsed["tool_name"] == "broken_tool"
    assert parsed["data"] is None

    assert parsed["error"]["type"] == (
        "ToolExecutionError"
    )

    assert parsed["error"]["message"] == (
        "Database connection failed"
    )

    print(
        "TEST: Tool Error Serialization PASS"
    )