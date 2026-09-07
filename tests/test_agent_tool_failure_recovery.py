from agents.agent import Agent


class FakeToolCall:
    type = "function_call"

    def __init__(
        self,
        name,
        call_id,
    ):
        self.name = name
        self.arguments = "{}"
        self.call_id = call_id


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
        self.messages = []

    def generate_with_tools(
        self,
        message,
        tools,
        instructions=None,
        previous_response_id=None,
    ):
        self.calls += 1
        self.messages.append(message)

        if self.calls == 1:
            return FakeResponse(
                output=[
                    FakeToolCall(
                        name="get_system_status",
                        call_id="call_status",
                    )
                ],
                response_id="response_1",
            )

        return FakeResponse(
            output=[],
            output_text="نتوانستم وضعیت سیستم را دریافت کنم.",
            response_id="response_2",
        )

class RecoveryLLM:
    def __init__(self):
        self.calls = 0
        self.messages = []

    def generate_with_tools(
        self,
        message,
        tools,
        instructions=None,
        previous_response_id=None,
    ):
        self.calls += 1
        self.messages.append(message)

        if self.calls == 1:
            return FakeResponse(
                output=[
                    FakeToolCall(
                        name="get_system_status",
                        call_id="call_status",
                    )
                ],
                response_id="response_1",
            )

        if self.calls == 2:
            return FakeResponse(
                output=[
                    FakeToolCall(
                        name="get_service_version",
                        call_id="call_version",
                    )
                ],
                response_id="response_2",
            )

        return FakeResponse(
            output=[],
            output_text="وضعیت و نسخه سرویس بررسی شد.",
            response_id="response_3",
        )


class MultiToolFailureLLM:
    def __init__(self):
        self.calls = 0
        self.messages = []

    def generate_with_tools(
        self,
        message,
        tools,
        instructions=None,
        previous_response_id=None,
    ):
        self.calls += 1
        self.messages.append(message)

        if self.calls == 1:
            return FakeResponse(
                output=[
                    FakeToolCall(
                        name="get_system_status",
                        call_id="call_status",
                    ),
                    FakeToolCall(
                        name="get_service_version",
                        call_id="call_version",
                    ),
                ],
                response_id="response_1",
            )

        return FakeResponse(
            output=[],
            output_text=(
                "وضعیت سیستم سالم است، اما "
                "دریافت نسخه سرویس ناموفق بود."
            ),
            response_id="response_2",
        )


def test_agent_recovers_from_tool_failure():
    llm = FakeLLM()

    executed_tools = []

    def fake_executor(
        tool_name,
        arguments,
    ):
        executed_tools.append(tool_name)

        if tool_name == "get_system_status":
            return {
                "success": False,
                "tool_name": tool_name,
                "data": None,
                "error": {
                    "type": "ToolExecutionError",
                    "message": "Database connection failed",
                },
            }

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

    result = agent.run(
        message="وضعیت سیستم را بررسی کن.",
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
            }
        ],
    )

    assert result == "نتوانستم وضعیت سیستم را دریافت کنم."

    assert executed_tools == [
        "get_system_status"
    ]

    assert llm.calls == 2

    assert llm.messages[1][0]["type"] == "function_call_output"

    assert "Database connection failed" in (
        llm.messages[1][0]["output"]
    )


def test_agent_recovers_by_selecting_another_tool():
    llm = RecoveryLLM()

    executed_tools = []

    def fake_executor(
        tool_name,
        arguments,
    ):
        executed_tools.append(tool_name)

        if tool_name == "get_system_status":
            return {
                "success": False,
                "tool_name": tool_name,
                "data": None,
                "error": {
                    "type": "ToolExecutionError",
                    "message": "Database connection failed",
                },
            }

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

    result = agent.run(
        message="وضعیت و نسخه سرویس را بررسی کن.",
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

    assert result == "وضعیت و نسخه سرویس بررسی شد."

    assert executed_tools == [
        "get_system_status",
        "get_service_version",
    ]

    assert llm.calls == 3

    assert "Database connection failed" in (
        llm.messages[1][0]["output"]
    )


def test_agent_preserves_successful_tool_result_when_another_tool_fails():
    llm = MultiToolFailureLLM()

    executed_tools = []

    def fake_executor(
        tool_name,
        arguments,
    ):
        executed_tools.append(tool_name)

        if tool_name == "get_service_version":
            return {
                "success": False,
                "tool_name": tool_name,
                "data": None,
                "error": {
                    "type": "ToolExecutionError",
                    "message": "Version service unavailable",
                },
            }

        return {
            "success": True,
            "tool_name": tool_name,
            "data": {
                "status": "healthy",
            },
            "error": None,
        }

    agent = Agent(
        llm_client=llm,
        tool_executor=fake_executor,
    )

    result = agent.run(
        message="وضعیت و نسخه سرویس را بررسی کن.",
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

    assert result == (
        "وضعیت سیستم سالم است، اما دریافت نسخه سرویس ناموفق بود."
    )

    assert executed_tools == [
        "get_system_status",
        "get_service_version",
    ]

    assert llm.calls == 2

    tool_outputs = llm.messages[1]

    assert len(tool_outputs) == 2

    assert "healthy" in tool_outputs[0]["output"]

    assert "Version service unavailable" in (
        tool_outputs[1]["output"]
    )