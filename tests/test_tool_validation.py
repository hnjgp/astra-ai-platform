from pydantic import ValidationError

from tools.executor import ToolExecutor


executor = ToolExecutor()


# ----------------------------------------
# Test 1: Valid arguments
# ----------------------------------------

result = executor.execute(
    tool_name="get_service_info",
    arguments={
        "service_name": "Astra",
    },
)

print("VALID RESULT:", result)

assert result["service"] == "Astra"
assert result["status"] == "available"

print("TEST 1: Valid Arguments PASS")


# ----------------------------------------
# Test 2: Missing required argument
# ----------------------------------------

try:

    executor.execute(
        tool_name="get_service_info",
        arguments={},
    )

    assert False, "Expected ValidationError"

except ValidationError:

    print("TEST 2: Missing Argument PASS")


# ----------------------------------------
# Test 3: Invalid argument type
# ----------------------------------------

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

# ----------------------------------------
# Test 4: Extra argument
# ----------------------------------------

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