# tests/test_system_tool.py

from tools.registry import get_tool


def test_system_status_tool():

    tool = get_tool(
        "get_system_status"
    )

    result = tool.execute()

    assert result["service"] == "Astra"
    assert result["status"] == "healthy"


def test_system_status_tool_name():

    tool = get_tool(
        "get_system_status"
    )

    assert tool.name == "get_system_status"


def test_system_status_tool_definition():

    tool = get_tool(
        "get_system_status"
    )

    definition = tool.to_definition()

    assert definition["type"] == "function"
    assert definition["name"] == "get_system_status"
    assert "description" in definition
    assert "parameters" in definition