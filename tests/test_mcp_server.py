import io
import json
import subprocess
import sys
from pathlib import Path

from mcp_server import MCPServer


def request(method, params=None, request_id=1):
    message = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
    }

    if params is not None:
        message["params"] = params

    return message


def test_initialize_negotiates_protocol_version():
    response = MCPServer().handle(
        request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        )
    )

    result = response["result"]

    assert result["protocolVersion"] == "2025-03-26"
    assert result["capabilities"] == {
        "tools": {"listChanged": False}
    }
    assert result["serverInfo"]["name"] == "astra"


def test_unknown_protocol_version_gets_latest():
    response = MCPServer().handle(
        request("initialize", {"protocolVersion": "1999-01-01"})
    )

    assert response["result"]["protocolVersion"] == "2025-06-18"


def test_initialized_notification_has_no_response():
    response = MCPServer().handle(
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
    )

    assert response is None


def test_tools_list_uses_mcp_shape():
    response = MCPServer().handle(request("tools/list"))

    tools = {
        tool["name"]: tool
        for tool in response["result"]["tools"]
    }

    assert set(tools) == {
        "get_system_status",
        "get_service_version",
        "get_service_info",
    }
    assert tools["get_service_info"]["inputSchema"]["required"] == [
        "service_name"
    ]


def test_tools_call_returns_text_and_structured_content():
    response = MCPServer().handle(
        request(
            "tools/call",
            {
                "name": "get_service_info",
                "arguments": {"service_name": "rag"},
            },
        )
    )

    result = response["result"]

    assert result["isError"] is False
    assert result["structuredContent"] == {
        "service": "rag",
        "status": "available",
    }
    assert json.loads(result["content"][0]["text"]) == (
        result["structuredContent"]
    )


def test_invalid_arguments_are_a_tool_error():
    response = MCPServer().handle(
        request(
            "tools/call",
            {
                "name": "get_service_info",
                "arguments": {},
            },
        )
    )

    assert response["result"]["isError"] is True


def test_unknown_tool_is_invalid_params():
    response = MCPServer().handle(
        request(
            "tools/call",
            {"name": "drop_database", "arguments": {}},
        )
    )

    assert response["error"]["code"] == -32602


def test_unknown_method():
    response = MCPServer().handle(request("resources/list"))

    assert response["error"]["code"] == -32601


def test_invalid_request():
    response = MCPServer().handle({"id": 5, "method": "ping"})

    assert response == {
        "jsonrpc": "2.0",
        "id": 5,
        "error": {"code": -32600, "message": "Invalid Request"},
    }


def test_serve_handles_parse_errors_and_skips_notifications():
    lines = "\n".join(
        [
            "{not json",
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                }
            ),
            json.dumps(request("ping", request_id=2)),
        ]
    )

    output = io.StringIO()

    MCPServer().serve(io.StringIO(lines), output)

    responses = [
        json.loads(line)
        for line in output.getvalue().splitlines()
    ]

    assert responses[0]["error"]["code"] == -32700
    assert responses[1] == {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {},
    }


def test_stdio_server_end_to_end():
    project_root = Path(__file__).resolve().parents[1]

    messages = [
        request("initialize", {"protocolVersion": "2025-06-18"}, 1),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        request(
            "tools/call",
            {"name": "get_system_status", "arguments": {}},
            2,
        ),
    ]

    completed = subprocess.run(
        [sys.executable, "-m", "mcp_server"],
        input="\n".join(json.dumps(m) for m in messages) + "\n",
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=60,
    )

    responses = [
        json.loads(line)
        for line in completed.stdout.splitlines()
    ]

    assert [response["id"] for response in responses] == [1, 2]
    assert responses[1]["result"]["structuredContent"] == {
        "service": "Astra",
        "status": "healthy",
    }
