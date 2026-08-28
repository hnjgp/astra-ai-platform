import pytest

from tools.executor import ToolExecutor


def test_unknown_tool_error():

    executor = ToolExecutor()

    with pytest.raises(KeyError):

        executor.execute(
            tool_name="unknown_tool",
            arguments={},
        )

    print("TEST: Unknown Tool Error PASS")