from pydantic import ValidationError

from tools.executor import ToolExecutor


executor = ToolExecutor()


def test_valid_arguments():

    result = executor.execute(
        tool_name="get_service_info",
        arguments={
            "service_name": "Astra",
        },
    )

    print("VALID RESULT:", result)

    assert result.success is True
    assert result.tool_name == "get_service_info"
    assert result.data["service"] == "Astra"
    assert result.data["status"] == "available"
    assert result.error is None

    print("TEST 1: Valid Arguments PASS")


def test_missing_required_argument():

    try:

        executor.execute(
            tool_name="get_service_info",
            arguments={},
        )

        assert False, "Expected ValidationError"

    except ValidationError:

        print("TEST 2: Missing Argument PASS")


def test_invalid_argument_type():

    try:

        executor.execute(
            tool_name="get_service_info",
            arguments={
                "service_name": 123,
            },
        )

        assert False, "Expected ValidationError"

    except ValidationError:

        print("TEST 3: Invalid Type PASS")


def test_extra_argument():

    try:

        executor.execute(
            tool_name="get_service_info",
            arguments={
                "service_name": "Astra",
                "unknown_field": "something",
            },
        )

        assert False, "Expected ValidationError"

    except ValidationError:

        print("TEST 4: Extra Argument PASS")