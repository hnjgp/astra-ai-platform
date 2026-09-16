from unittest.mock import Mock

from agents.agent import Agent


def test_agent_uses_rag_prompt_when_rag_is_enabled():
    llm_client = Mock()
    tool_executor = Mock()
    rag_service = Mock()

    response = Mock()
    response.id = "response-1"
    response.output = []
    response.output_text = "Answer from knowledge base."

    llm_client.generate_with_tools.return_value = response

    rag_service.build_prompt.return_value = (
        "Use the knowledge base context to answer the question."
    )

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
        rag_service=rag_service,
    )

    answer = agent.run(
        message="How does authentication work?",
        tools=[],
        instructions="You are Astra.",
        use_rag=True,
    )

    assert answer == "Answer from knowledge base."

    rag_service.build_prompt.assert_called_once_with(
        question="How does authentication work?",
    )

    llm_client.generate_with_tools.assert_called_once_with(
        message=[
            {
                "role": "user",
                "content": "How does authentication work?",
            }
        ],
        tools=[],
        instructions=(
            "You are Astra.\n\n"
            "Use the knowledge base context to answer the question."
        ),
    )


def test_agent_does_not_use_rag_when_rag_is_disabled():
    llm_client = Mock()
    tool_executor = Mock()
    rag_service = Mock()

    response = Mock()
    response.id = "response-1"
    response.output = []
    response.output_text = "Normal answer."

    llm_client.generate_with_tools.return_value = response

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
        rag_service=rag_service,
    )

    answer = agent.run(
        message="What is Python?",
        tools=[],
        instructions="You are Astra.",
        use_rag=False,
    )

    assert answer == "Normal answer."

    rag_service.build_prompt.assert_not_called()

    llm_client.generate_with_tools.assert_called_once_with(
        message=[
            {
                "role": "user",
                "content": "What is Python?",
            }
        ],
        tools=[],
        instructions="You are Astra.",
    )


def test_agent_can_use_rag_without_existing_instructions():
    llm_client = Mock()
    tool_executor = Mock()
    rag_service = Mock()

    response = Mock()
    response.id = "response-1"
    response.output = []
    response.output_text = "RAG answer."

    llm_client.generate_with_tools.return_value = response

    rag_service.build_prompt.return_value = (
        "Use only the provided knowledge base."
    )

    agent = Agent(
        llm_client=llm_client,
        tool_executor=tool_executor,
        rag_service=rag_service,
    )

    answer = agent.run(
        message="What does the documentation say?",
        tools=[],
        use_rag=True,
    )

    assert answer == "RAG answer."

    llm_client.generate_with_tools.assert_called_once_with(
        message=[
            {
                "role": "user",
                "content": "What does the documentation say?",
            }
        ],
        tools=[],
        instructions=(
            "Use only the provided knowledge base."
        ),
    )