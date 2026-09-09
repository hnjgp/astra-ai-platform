# agents/memory.py

from typing import Any


class Memory:
    """
    Simple memory abstraction for the agent.

    This first implementation keeps memories
    in memory during the current application process.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def save(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Store a value under a key.
        """

        self._store[key] = value

    def retrieve(
        self,
        key: str,
    ) -> Any | None:
        """
        Retrieve a value by key.

        Returns None when the key does not exist.
        """

        return self._store.get(key)