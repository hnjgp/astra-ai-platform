# agents/memory_manager.py

from typing import Any

from agents.memory import Memory


class MemoryManager:
    """
    Controls how the agent reads from and writes to memory.

    The manager keeps memory-related decisions separate
    from the Memory storage implementation.
    """

    def __init__(
        self,
        memory: Memory | None = None,
    ) -> None:

        self.memory = (
            memory
            if memory is not None
            else Memory()
        )

    def save(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Save a value into memory.
        """

        self.memory.save(
            key,
            value,
        )

    def retrieve(
        self,
        key: str,
    ) -> Any | None:
        """
        Retrieve a value from memory.

        Returns None when the memory does not exist.
        """

        return self.memory.retrieve(
            key
        )

    def build_context(
        self,
        key: str,
    ) -> list[dict]:
        """
        Build model context from a memory value.

        Returns an empty list when the requested
        memory does not exist.
        """

        value = self.retrieve(
            key
        )

        if value is None:
            return []

        return [
            {
                "role": "system",
                "content": (
                    "Relevant memory:\n"
                    f"{value}"
                ),
            }
        ]