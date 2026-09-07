from agents.agent import Agent
import pytest

class FakeToolCall:
    type = "function_call"
    name = "get_service_version"
    arguments = "{}"
    call_id = "call_version"


class FakeResponse:
    def __init__(
        self,
        output,
        output_text="",
        response_id="response_1",
    ):
        self.output = output
        self.output_text = output_text
        self.id = response_id


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate_with_tools(
        self,
        message,
        tools,
        instructions=None,
        previous_response_id=None,
    ):
        self.calls += 1

        if self.calls == 1:
            return FakeResponse(
                output=[
                    FakeToolCall()
                ],
                response_id="response_1",
            )

        return FakeResponse(
            output=[],
            output_text="نسخه سرویس ۱.۰.۰ است.",
            response_id="response_2",
        )


def test_agent_executes_selected_tool_only():
    llm = FakeLLM()

    executed_tools = []

    def fake_executor(
        tool_name,
        arguments,
    ):
        executed_tools.append(
            tool_name
        )

        return {
            "success": True,
            "tool_name": tool_name,
            "data": {
                "version": "1.0.0",
            },
            "error": None,
        }

    agent = Agent(
        llm_client=llm,
        tool_executor=fake_executor,
    )

    tools = [
        {
            "type": "function",
            "name": "get_system_status",
            "description": "Get system status.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "get_service_version",
            "description": "Get service version.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    ]

    result = agent.run(
        message="نسخه سرویس را بگو.",
        tools=tools,
    )

    assert result == "نسخه سرویس ۱.۰.۰ است."

    assert executed_tools == [
        "get_service_version"
    ]





class UnknownToolCall:
    type = "function_call"
    name = "unknown_tool"
    arguments = "{}"
    call_id = "call_unknown"


class UnknownToolLLM:
    def __init__(self):
        self.calls = 0

    def generate_with_tools(
        self,
        message,
        tools,
        instructions=None,
        previous_response_id=None,
    ):
        self.calls += 1

        return FakeResponse(
            output=[
                UnknownToolCall()
            ],
            response_id=f"response_{self.calls}",
        )


def test_agent_handles_unknown_selected_tool():
    llm = UnknownToolLLM()

    def fake_executor(
        tool_name,
        arguments,
    ):
        raise KeyError(
            f"Unknown tool: {tool_name}"
        )

    agent = Agent(
        llm_client=llm,
        tool_executor=fake_executor,
    )

    with pytest.raises(KeyError):
        agent.run(
            message="یک ابزار ناشناخته را اجرا کن.",
            tools=[],
            max_tool_rounds=1,
        )

class MultiToolLLM:
    def __init__(self):
        self.calls = 0

    def generate_with_tools(
        self,
        message,
        tools,
        instructions=None,
        previous_response_id=None,
    ):
        self.calls += 1

        if self.calls == 1:
            return FakeResponse(
                output=[
                    type(
                        "FakeToolCall",
                        (),
                        {
                            "type": "function_call",
                            "name": "get_system_status",
                            "arguments": "{}",
                            "call_id": "call_status",
                        },
                    )(),
                    type(
                        "FakeToolCall",
                        (),
                        {
                            "type": "function_call",
                            "name": "get_service_version",
                            "arguments": "{}",
                            "call_id": "call_version",
                        },
                    )(),
                ],
                response_id="response_1",
            )

        return FakeResponse(
            output=[],
            output_text="وضعیت سالم است و نسخه ۱.۰.۰ است.",
            response_id="response_2",
        )


def test_agent_executes_multiple_selected_tools():
    llm = MultiToolLLM()

    executed_tools = []

    def fake_executor(
        tool_name,
        arguments,
    ):
        executed_tools.append(tool_name)

        return {
            "success": True,
            "tool_name": tool_name,
            "data": {},
            "error": None,
        }

    agent = Agent(
        llm_client=llm,
        tool_executor=fake_executor,
    )

    result = agent.run(
        message="وضعیت سیستم و نسخه سرویس را بررسی کن.",
        tools=[
            {
                "type": "function",
                "name": "get_system_status",
                "description": "Get system status.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_service_version",
                "description": "Get service version.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        ],
    )

    assert result == "وضعیت سالم است و نسخه ۱.۰.۰ است."

    assert executed_tools == [
        "get_system_status",
        "get_service_version",
    ]

    assert llm.calls == 2

def test_agent_preserves_selected_tool_order():
    llm = MultiToolLLM()

    executed_tools = []

    def fake_executor(
        tool_name,
        arguments,
    ):
        executed_tools.append(tool_name)

        return {
            "success": True,
            "tool_name": tool_name,
            "data": {},
            "error": None,
        }

    agent = Agent(
        llm_client=llm,
        tool_executor=fake_executor,
    )

    agent.run(
        message="وضعیت و نسخه را بررسی کن.",
        tools=[
            {
                "type": "function",
                "name": "get_system_status",
                "description": "Get system status.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_service_version",
                "description": "Get service version.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        ],
    )

    assert executed_tools == [
        "get_system_status",
        "get_service_version",
    ]