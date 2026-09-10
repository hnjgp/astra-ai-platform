# tests/test_agent_memory.py

from agents.agent import Agent
from agents.memory import Memory
from agents.memory_manager import MemoryManager


class FakeResponse:

    def __init__(
        self,
        output_text="",
        response_id="response_1",
    ):
        self.output = []
        self.output_text = output_text
        self.id = response_id


class FakeLLM:

    def __init__(self):

        self.calls = []

    def generate_with_tools(
        self,
        message,
        tools,
        instructions=None,
        previous_response_id=None,
    ):

        self.calls.append(
            {
                "message": message,
                "tools": tools,
                "instructions": instructions,
                "previous_response_id": previous_response_id,
            }
        )

        return FakeResponse(
            output_text="پاسخ نهایی",
        )


def test_agent_runs_without_memory():

    llm = FakeLLM()

    agent = Agent(
        llm_client=llm,
        tool_executor=lambda **kwargs: None,
    )

    result = agent.run(
        message="سلام",
        tools=[],
    )

    assert result == "پاسخ نهایی"

    assert llm.calls[0]["message"] == [
        {
            "role": "user",
            "content": "سلام",
        }
    ]


def test_agent_reads_selected_memory():

    memory = Memory()

    memory.save(
        "user_goal",
        "AI Engineer",
    )

    memory_manager = MemoryManager(
        memory=memory
    )

    llm = FakeLLM()

    agent = Agent(
        llm_client=llm,
        tool_executor=lambda **kwargs: None,
        memory_manager=memory_manager,
    )

    result = agent.run(
        message="برای چه شغلی آماده می‌شوم؟",
        tools=[],
        memory_key="user_goal",
    )

    assert result == "پاسخ نهایی"

    assert llm.calls[0]["message"] == [
        {
            "role": "system",
            "content": (
                "Relevant memory:\n"
                "AI Engineer"
            ),
        },
        {
            "role": "user",
            "content": "برای چه شغلی آماده می‌شوم؟",
        },
    ]


def test_agent_ignores_memory_when_no_memory_key_is_provided():

    memory = Memory()

    memory.save(
        "user_goal",
        "AI Engineer",
    )

    memory_manager = MemoryManager(
        memory=memory
    )

    llm = FakeLLM()

    agent = Agent(
        llm_client=llm,
        tool_executor=lambda **kwargs: None,
        memory_manager=memory_manager,
    )

    agent.run(
        message="سلام",
        tools=[],
    )

    assert llm.calls[0]["message"] == [
        {
            "role": "user",
            "content": "سلام",
        }
    ]


def test_agent_handles_missing_memory():

    memory = Memory()

    memory_manager = MemoryManager(
        memory=memory
    )

    llm = FakeLLM()

    agent = Agent(
        llm_client=llm,
        tool_executor=lambda **kwargs: None,
        memory_manager=memory_manager,
    )

    result = agent.run(
        message="سلام",
        tools=[],
        memory_key="unknown",
    )

    assert result == "پاسخ نهایی"

    assert llm.calls[0]["message"] == [
        {
            "role": "user",
            "content": "سلام",
        }
    ]


def test_agent_uses_only_selected_memory():

    memory = Memory()

    memory.save(
        "user_goal",
        "AI Engineer",
    )

    memory.save(
        "learning_level",
        "beginner",
    )

    memory_manager = MemoryManager(
        memory=memory
    )

    llm = FakeLLM()

    agent = Agent(
        llm_client=llm,
        tool_executor=lambda **kwargs: None,
        memory_manager=memory_manager,
    )

    agent.run(
        message="سلام",
        tools=[],
        memory_key="user_goal",
    )

    assert llm.calls[0]["message"] == [
        {
            "role": "system",
            "content": (
                "Relevant memory:\n"
                "AI Engineer"
            ),
        },
        {
            "role": "user",
            "content": "سلام",
        },
    ]