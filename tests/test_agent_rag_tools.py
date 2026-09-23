from unittest.mock import Mock

from agents.agent import Agent


def test_agent_uses_rag_and_tools_in_the_same_run():
    llm_client = Mock()
    tool_executor = Mock()
    rag_service = Mock()

    first_response = Mock()
    first_response.id = "response-1"

    tool_call = Mock()
    tool_call.type = "function_call"
    tool_call.name = "get_system_status"
    tool_call.arguments = "{}"
    tool_call.call_id = "call-1"

    first_response.output = [tool_call]
    first_response.output_text = ""

    second_response = Mock()
    second_response.id = "response-2"
    second_response.output = []
    second_response.output_text = (
        "The system is healthy."
    )

    llm_client.generate_with_tools.side_effect = [
        first_response,
        second_response,
    ]

    rag_service.build_prompt.return_value = (
        "Use the knowledge base context."
    )

    tool_executor.return_value = {
        "status": "healthy"
    }

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
        rag_service=rag_service,
    )

    answer = agent.run(
        message="What is the current system status?",
        tools=[
            {
                "type": "function",
                "name": "get_system_status",
            }
        ],
        instructions="You are Astra.",
        use_rag=True,
    )

    assert answer == "The system is healthy."

    rag_service.build_prompt.assert_called_once_with(
        question="What is the current system status?",
    )

    tool_executor.assert_called_once_with(
        tool_name="get_system_status",
        arguments="{}",
    )

    assert llm_client.generate_with_tools.call_count == 2

    first_call = (
        llm_client.generate_with_tools.call_args_list[0]
    )

    assert first_call.kwargs["instructions"] == (
        "You are Astra.\n\n"
        "Use the knowledge base context."
    )

    second_call = (
        llm_client.generate_with_tools.call_args_list[1]
    )

    assert second_call.kwargs["instructions"] == (
        "You are Astra.\n\n"
        "Use the knowledge base context."
    )

    second_message = second_call.kwargs["message"]

    assert any(
        item.get("type") == "function_call_output"
        and item.get("call_id") == "call-1"
        and "healthy" in item.get("output", "")
        for item in second_message
    )