from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.postgres import (
    PostgresSaver,
)


def to_psycopg_url(database_url: str) -> str:
    """
    PostgresSaver uses psycopg directly, which does not understand the
    SQLAlchemy driver suffix (postgresql+psycopg://).
    """

    return database_url.replace(
        "postgresql+psycopg://",
        "postgresql://",
        1,
    )


@contextmanager
def postgres_checkpointer(
    database_url: str,
) -> Iterator[PostgresSaver]:

    with PostgresSaver.from_conn_string(
        to_psycopg_url(database_url)
    ) as checkpointer:

        checkpointer.setup()

        yield checkpointer