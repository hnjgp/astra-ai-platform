from unittest.mock import Mock

import pytest

from agents.planner import Planner
from exceptions import LLMError


def test_planner_creates_valid_plan():

    llm_client = Mock()

    llm_client.generate.return_value = """
    {
        "steps": [
            {
                "step": 1,
                "description": "Check system status"
            },
            {
                "step": 2,
                "description": "Explain the result"
            }
        ]
    }
    """

    planner = Planner(
        llm_client=llm_client
    )

    plan = planner.create_plan(
        "Check the system status and explain the result."
    )

    assert len(plan.steps) == 2

    assert plan.steps[0].step == 1
    assert (
        plan.steps[0].description
        == "Check system status"
    )

    assert plan.steps[1].step == 2
    assert (
        plan.steps[1].description
        == "Explain the result"
    )

    llm_client.generate.assert_called_once()


def test_planner_rejects_invalid_json():

    llm_client = Mock()

    llm_client.generate.return_value = (
        "this is not valid json"
    )

    planner = Planner(
        llm_client=llm_client
    )

    with pytest.raises(LLMError):
        planner.create_plan(
            "Check the system status."
        )


def test_planner_rejects_invalid_plan():

    llm_client = Mock()

    llm_client.generate.return_value = """
    {
        "wrong_field": "wrong_value"
    }
    """

    planner = Planner(
        llm_client=llm_client
    )

    with pytest.raises(LLMError):
        planner.create_plan(
            "Check the system status."
        )