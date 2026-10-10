"""
Minimal MCP (Model Context Protocol) server for Astra's tools.

Transport: stdio, one JSON-RPC 2.0 message per line.
Implements: initialize, notifications/initialized, ping,
tools/list and tools/call.

Run:
    python -m mcp_server
"""

import json
import logging
import sys
from typing import Any, TextIO

from pydantic import ValidationError

from tools.executor import ToolExecutor


SERVER_INFO = {
    "name": "astra",
    "version": "1.0.0",
}

SUPPORTED_PROTOCOL_VERSIONS = (
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)

# MCP clients act with the lowest role: only read-only system tools.
MCP_ROLE = "guest"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

# Logs must go to stderr; stdout carries protocol messages only.
logger = logging.getLogger("astra.mcp")


class MCPServer:

    def __init__(
        self,
        executor: ToolExecutor | None = None,
        role: str = MCP_ROLE,
    ):
        self.executor = executor or ToolExecutor()
        self.role = role

    def handle(
        self,
        message: Any,
    ) -> dict | None:
        """
        Handle one JSON-RPC message. Notifications return None.
        """

        if (
            not isinstance(message, dict)
            or message.get("jsonrpc") != "2.0"
            or not isinstance(message.get("method"), str)
        ):
            request_id = (
                message.get("id")
                if isinstance(message, dict)
                else None
            )

            return self._error(
                request_id,
                INVALID_REQUEST,
                "Invalid Request",
            )

        method = message["method"]
        params = message.get("params") or {}

        if "id" not in message:
            # Notifications (e.g. notifications/initialized) get no reply.
            return None

        request_id = message["id"]

        handlers = {
            "initialize": self._initialize,
            "ping": lambda _: {},
            "tools/list": self._list_tools,
            "tools/call": self._call_tool,
        }

        handler = handlers.get(method)

        if handler is None:
            return self._error(
                request_id,
                METHOD_NOT_FOUND,
                f"Method not found: {method}",
            )

        try:
            result = handler(params)

        except _InvalidParams as exc:
            return self._error(
                request_id,
                INVALID_PARAMS,
                str(exc),
            )

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

    def _initialize(self, params: dict) -> dict:
        requested = params.get("protocolVersion")

        version = (
            requested
            if requested in SUPPORTED_PROTOCOL_VERSIONS
            else SUPPORTED_PROTOCOL_VERSIONS[0]
        )

        return {
            "protocolVersion": version,
            "capabilities": {
                "tools": {
                    "listChanged": False,
                },
            },
            "serverInfo": SERVER_INFO,
        }

    def _list_tools(self, params: dict) -> dict:
        return {
            "tools": [
                {
                    "name": definition["name"],
                    "description": definition["description"],
                    "inputSchema": definition["parameters"],
                }
                for definition in self.executor.tool_definitions(
                    self.role
                )
            ]
        }

    def _call_tool(self, params: dict) -> dict:
        name = params.get("name")
        arguments = params.get("arguments") or {}

        allowed = {
            definition["name"]
            for definition in self.executor.tool_definitions(
                self.role
            )
        }

        if name not in allowed:
            raise _InvalidParams(f"Unknown tool: {name}")

        try:
            result = self.executor.execute(
                tool_name=name,
                arguments=arguments,
                role=self.role,
            )

        except (ValidationError, ValueError) as exc:
            # Bad arguments are a tool error the model can correct,
            # not a protocol error.
            return self._tool_error(str(exc))

        if not result.success:
            return self._tool_error(
                result.error.message
                if result.error
                else "Tool execution failed"
            )

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        result.data,
                        ensure_ascii=False,
                    ),
                }
            ],
            "structuredContent": result.data,
            "isError": False,
        }

    @staticmethod
    def _tool_error(message: str) -> dict:
        return {
            "content": [
                {
                    "type": "text",
                    "text": message,
                }
            ],
            "isError": True,
        }

    @staticmethod
    def _error(
        request_id: Any,
        code: int,
        message: str,
    ) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }

    def serve(
        self,
        input_stream: TextIO,
        output_stream: TextIO,
    ) -> None:
        for line in input_stream:
            if not line.strip():
                continue

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                response = self._error(
                    None,
                    PARSE_ERROR,
                    "Parse error",
                )
            else:
                response = self.handle(message)

            if response is None:
                continue

            output_stream.write(
                json.dumps(response, ensure_ascii=False) + "\n"
            )
            output_stream.flush()


class _InvalidParams(Exception):
    pass


def main() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
    )

    logger.info("Astra MCP server started on stdio")

    MCPServer().serve(
        sys.stdin,
        sys.stdout,
    )


if __name__ == "__main__":
    main()
