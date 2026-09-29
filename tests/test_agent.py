from unittest.mock import Mock

from agents.agent import Agent
from agents.planner import Planner


class FakeLLM:

    def __init__(self):
        self.generate_with_tools_calls = []

    def generate_with_tools(
        self,
        message,
        tools,
        instructions=None,
        previous_response_id=None,
    ):

        self.generate_with_tools_calls.append(
            {
                "message": message,
                "tools": tools,
                "instructions": instructions,
                "previous_response_id": (
                    previous_response_id
                ),
            }
        )

        class Response:

            output = []

            output_text = "پاسخ مستقیم Agent"

            id = "response_1"

        return Response()


def test_agent_without_tools():

    agent = Agent(
        llm_client=FakeLLM(),
        tool_executor=lambda **kwargs: None,
    )

    result = agent.run(
        message="سلام",
        tools=[],
    )

    assert result == "پاسخ مستقیم Agent"


def test_agent_uses_planner_when_planning_enabled():

    llm_client = FakeLLM()

    planner = Mock(
        spec=Planner
    )

    class FakePlan:

        def model_dump_json(
            self,
            indent=None,
            ensure_ascii=False,
        ):
            return (
                '{'
                '"steps": ['
                '{'
                '"step": 1, '
                '"description": '
                '"Check system status"'
                '}'
                ']'
                '}'
            )

    planner.create_plan.return_value = (
        FakePlan()
    )

    agent = Agent(
        llm_client=llm_client,
        tool_executor=lambda **kwargs: None,
        planner=planner,
    )

    result = agent.run(
        message="وضعیت سیستم را بررسی کن",
        tools=[],
        planning=True,
    )

    assert result == "پاسخ مستقیم Agent"

    planner.create_plan.assert_called_once_with(
        "وضعیت سیستم را بررسی کن"
    )

    assert (
        "Execution plan:"
        in llm_client.generate_with_tools_calls[
            0
        ]["instructions"]
    )

    assert (
        "Check system status"
        in llm_client.generate_with_tools_calls[
            0
        ]["instructions"]
    )


def test_agent_requires_planner_when_planning_enabled():

    agent = Agent(
        llm_client=FakeLLM(),
        tool_executor=lambda **kwargs: None,
    )

    try:
        agent.run(
            message="وضعیت سیستم را بررسی کن",
            tools=[],
            planning=True,
        )
    except ValueError as exc:
        assert (
            str(exc)
            == "planner is required when planning is True"
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )