from tools.executor import ToolExecutor


executor = ToolExecutor()


try:
    executor.execute(
        tool_name="unknown_tool",
        arguments={},
    )
    assert False, "Expected KeyError for unknown tool"

except KeyError:
    print("TEST: Unknown Tool Error PASS")