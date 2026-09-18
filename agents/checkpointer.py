from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.postgres import (
    PostgresSaver,
)


@contextmanager
def postgres_checkpointer(
    database_url: str,
) -> Iterator[PostgresSaver]:

    with PostgresSaver.from_conn_string(
        database_url
    ) as checkpointer:

        checkpointer.setup()

        yield checkpointer