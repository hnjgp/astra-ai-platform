from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel


class BaseTool(ABC):
    """
    Base abstraction for all Astra tools.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique tool name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the tool description."""
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Return the tool input schema for the LLM."""
        raise NotImplementedError

    @property
    def input_schema(self) -> Type[BaseModel] | None:
        """
        Return the Pydantic model used to validate tool arguments.
        """
        return None

    def to_definition(self) -> dict[str, Any]:
        """
        Return the complete tool definition for the LLM.
        """
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool."""
        raise NotImplementedError