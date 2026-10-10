from unittest.mock import Mock

import pytest

from agents.agent import Agent
from agents.planner import Planner
from exceptions import LLMError
from tests.fakes import ScriptedLLMClient


TOOLS = [
    {
        "type": "function",
        "name": "get_system_status",
        "description": "Get system status",
        "parameters": {},
    }
]


def test_failing_tool_cannot_loop_forever():
    llm = ScriptedLLMClient(
        [[("get_system_status", "{}")]] * 50
    )

    tool_executor = Mock(
        side_effect=TimeoutError("tool timed out")
    )

    agent = Agent(
        llm_client=llm,
        tool_executor=tool_executor,
    )

    with pytest.raises(
        LLMError,
        match="Maximum tool execution rounds exceeded",
    ):
        agent.run(
            message="Check status",
            tools=TOOLS,
            max_tool_rounds=3,
        )

    # Each failed round counts against max_tool_rounds.
    assert tool_executor.call_count == 3


def test_recovered_run_is_reported():
    llm = ScriptedLLMClient(
        [
            [("get_system_status", "{}")],
            "The status tool is unavailable right now.",
        ]
    )

    agent = Agent(
        llm_client=llm,
        tool_executor=Mock(
            side_effect=ConnectionError("down")
        ),
    )

    result = agent.run_detailed(
        message="Check status",
        tools=TOOLS,
    )

    assert result.recovered is True
    assert result.tools_used == []
    assert "Tool execution failed" in (
        llm.calls[1]["message"][0]["output"]
    )


def test_recursion_limit_becomes_llm_error():
    llm = ScriptedLLMClient(
        [[("get_system_status", "{}")]] * 50
    )

    agent = Agent(
        llm_client=llm,
        tool_executor=Mock(return_value={"status": "ok"}),
    )

    agent.recursion_limit = 3

    with pytest.raises(
        LLMError,
        match="step limit",
    ):
        agent.run(
            message="Check status",
            tools=TOOLS,
            max_tool_rounds=20,
        )


def test_planner_accepts_json_inside_markdown_fence():
    llm = Mock()
    llm.generate.return_value = (
        "```json\n"
        '{"steps": [{"step": 1, "description": "Check status"}]}\n'
        "```"
    )

    plan = Planner(llm).create_plan("Check status")

    assert plan.steps[0].description == "Check status"


def test_planner_rejects_too_many_steps():
    steps = ", ".join(
        f'{{"step": {number}, "description": "step {number}"}}'
        for number in range(1, 13)
    )

    llm = Mock()
    llm.generate.return_value = f'{{"steps": [{steps}]}}'

    with pytest.raises(
        LLMError,
        match="too many steps",
    ):
        Planner(llm).create_plan("Do many things")


def test_planner_rejects_out_of_order_steps():
    llm = Mock()
    llm.generate.return_value = (
        '{"steps": ['
        '{"step": 2, "description": "b"}, '
        '{"step": 1, "description": "a"}'
        "]}"
    )

    with pytest.raises(
        LLMError,
        match="invalid plan",
    ):
        Planner(llm).create_plan("Do things")
