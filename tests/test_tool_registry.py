from tools.base import BaseTool
from tools.registry import (
    get_tool,
    get_tool_definitions,
)


def test_registry_returns_tool_object():
    tool = get_tool("get_system_status")

    assert isinstance(tool, BaseTool)
    assert tool.name == "get_system_status"


def test_registry_returns_all_tool_definitions():
    definitions = get_tool_definitions()

    assert isinstance(definitions, list)

    names = {
        definition["name"]
        for definition in definitions
    }

    assert "get_system_status" in names
    assert "get_service_version" in names


def test_tool_definitions_have_valid_structure():
    definitions = get_tool_definitions()

    for definition in definitions:

        assert definition["type"] == "function"
        assert isinstance(definition["name"], str)
        assert isinstance(definition["description"], str)

        parameters = definition["parameters"]

        assert parameters["type"] == "object"
        assert "properties" in parameters
        assert "required" in parameters
        assert parameters["additionalProperties"] is False


def test_unknown_tool_raises_key_error():
    try:
        get_tool("unknown_tool")
        assert False, "Expected KeyError"
    except KeyError:
        pass