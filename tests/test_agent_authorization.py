from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.agent import Agent
from tools.authorization import ToolAuthorization
from tools.executor import ToolExecutor
from tools.guardrail import ToolGuardrail


def make_tool_response(
    response_id: str,
    tool_name: str,
):

    tool_call = SimpleNamespace(
        type="function_call",
        call_id="call_1",
        name=tool_name,
        arguments="{}",
    )

    return SimpleNamespace(
        id=response_id,
        output=[tool_call],
        output_text=None,
    )


def test_agent_passes_role_to_tool_authorization():

    llm_client = Mock()

    llm_client.generate_with_tools.return_value = (
        make_tool_response(
            response_id="response_1",
            tool_name="get_system_status",
        )
    )

    authorization = ToolAuthorization(
        role_permissions={
            "admin": set(),
        },
    )

    executor = ToolExecutor(
        guardrail=ToolGuardrail(),
        authorization=authorization,
    )

    agent = Agent(
        llm_client=llm_client,
        tool_executor=executor.execute,
    )

    with pytest.raises(
        ValueError,
        match="Tool is not allowed for role",
    ):
        agent.run(
            message="What is the system status?",
            tools=[
                {
                    "type": "function",
                    "name": "get_system_status",
                    "description": "Get system status",
                    "parameters": {},
                }
            ],
            role="admin",
        )

    llm_client.generate_with_tools.assert_called_once()