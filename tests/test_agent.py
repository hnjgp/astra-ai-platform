from unittest.mock import Mock

import pytest

from agents.agent import Agent
from exceptions import LLMError
from schemas import Plan, PlanStep


def test_agent_without_tools():
    llm_client = Mock()
    llm_client.generate_with_tools.return_value = Mock(
        id="response_1",
        output=[],
        output_text="Hello from Astra",
    )

    tool_executor = Mock()

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
    )

    result = agent.run(
        message="Hello",
        tools=[],
    )

    assert result == "Hello from Astra"

    llm_client.generate_with_tools.assert_called_once()

    tool_executor.assert_not_called()


def test_agent_uses_planner_when_planning_enabled():
    llm_client = Mock()
    llm_client.generate.return_value = "unused"

    llm_client.generate_with_tools.return_value = Mock(
        id="response_1",
        output=[],
        output_text="Planned response",
    )

    tool_executor = Mock()

    planner = Mock()

    planner.create_plan.return_value = Plan(
        steps=[
            PlanStep(
                step=1,
                description="Check system status",
            )
        ]
    )

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
        planner=planner,
    )

    result = agent.run(
        message="Check the system status",
        tools=[],
        planning=True,
    )

    assert result == "Planned response"

    planner.create_plan.assert_called_once_with(
        "Check the system status"
    )

    llm_client.generate_with_tools.assert_called_once()

    instructions = (
        llm_client.generate_with_tools.call_args.kwargs[
            "instructions"
        ]
    )

    assert "Execution plan:" in instructions
    assert "Check system status" in instructions


def test_agent_requires_planner_when_planning_enabled():
    llm_client = Mock()
    tool_executor = Mock()

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
    )

    with pytest.raises(
        ValueError,
        match="planner is required when planning is True",
    ):
        agent.run(
            message="Check the system status",
            tools=[],
            planning=True,
        )


def test_agent_planning_rejects_empty_plan():
    llm_client = Mock()
    tool_executor = Mock()

    planner = Mock()

    planner.create_plan.return_value = Plan(
        steps=[]
    )

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
        planner=planner,
    )

    with pytest.raises(
        ValueError,
        match="planner returned an empty plan",
    ):
        agent.run(
            message="Check the system status",
            tools=[],
            planning=True,
        )


def test_agent_preserves_existing_instructions_with_plan():
    llm_client = Mock()

    llm_client.generate_with_tools.return_value = Mock(
        id="response_1",
        output=[],
        output_text="Final answer",
    )

    tool_executor = Mock()

    planner = Mock()

    planner.create_plan.return_value = Plan(
        steps=[
            PlanStep(
                step=1,
                description="Check system status",
            )
        ]
    )

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
        planner=planner,
    )

    result = agent.run(
        message="Check the system status",
        tools=[],
        instructions="Answer clearly.",
        planning=True,
    )

    assert result == "Final answer"

    instructions = (
        llm_client.generate_with_tools.call_args.kwargs[
            "instructions"
        ]
    )

    assert "Answer clearly." in instructions
    assert "Execution plan:" in instructions
    assert "Check system status" in instructions


def test_agent_planning_requires_non_empty_plan():
    llm_client = Mock()
    tool_executor = Mock()

    planner = Mock()

    planner.create_plan.return_value = Plan(
        steps=[]
    )

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
        planner=planner,
    )

    with pytest.raises(
        ValueError,
        match="planner returned an empty plan",
    ):
        agent.run(
            message="Do something",
            tools=[],
            planning=True,
        )