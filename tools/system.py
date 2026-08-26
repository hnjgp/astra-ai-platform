from typing import Any

from pydantic import BaseModel

from schemas import ServiceInfoInput
from tools.base import BaseTool


class SystemStatusTool(BaseTool):

    @property
    def name(self) -> str:
        return "get_system_status"

    @property
    def description(self) -> str:
        return "Get the current health status of Astra."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def execute(self, **kwargs: Any) -> dict[str, str]:
        return {
            "service": "Astra",
            "status": "healthy",
        }


class ServiceVersionTool(BaseTool):

    @property
    def name(self) -> str:
        return "get_service_version"

    @property
    def description(self) -> str:
        return "Get the current version of the Astra service."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def execute(self, **kwargs: Any) -> dict[str, str]:
        return {
            "service": "Astra",
            "version": "1.0.0",
        }


class ServiceInfoTool(BaseTool):

    @property
    def name(self) -> str:
        return "get_service_info"

    @property
    def description(self) -> str:
        return "Get information about an Astra service."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "Name of the service.",
                }
            },
            "required": ["service_name"],
            "additionalProperties": False,
        }

    @property
    def input_schema(self) -> type[BaseModel]:
        return ServiceInfoInput

    def execute(self, **kwargs: Any) -> dict[str, str]:
        data = ServiceInfoInput(**kwargs)

        return {
            "service": data.service_name,
            "status": "available",
        }