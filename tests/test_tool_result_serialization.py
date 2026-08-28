import json

from schemas import ToolResult


def test_tool_result_serialization():

    result = ToolResult(
        success=True,
        tool_name="get_system_status",
        data={
            "service": "Astra",
            "status": "healthy",
        },
        error=None,
    )

    print("PYDANTIC RESULT:")
    print(result)

    dumped = result.model_dump()

    print("DUMP RESULT:")
    print(dumped)

    assert dumped["success"] is True
    assert dumped["tool_name"] == "get_system_status"
    assert dumped["data"]["service"] == "Astra"
    assert dumped["data"]["status"] == "healthy"
    assert dumped["error"] is None

    json_result = result.model_dump_json()

    print("JSON RESULT:")
    print(json_result)

    parsed = json.loads(json_result)

    assert parsed["success"] is True
    assert parsed["tool_name"] == "get_system_status"
    assert parsed["data"]["service"] == "Astra"
    assert parsed["data"]["status"] == "healthy"
    assert parsed["error"] is None

    print(
        "TEST: Tool Result Serialization PASS"
    )