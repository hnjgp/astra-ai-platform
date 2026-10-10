from sqlalchemy import text

from rag.database import init_rag_database, rag_engine


def test_rag_database_init_is_idempotent():
    init_rag_database()
    init_rag_database()


def test_rag_indexes_exist():
    init_rag_database()

    with rag_engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'document_chunks'
                """
            )
        ).all()

    definitions = {name: definition for name, definition in rows}

    assert "ix_document_chunks_document_id" in definitions
    assert "hnsw" in definitions["ix_document_chunks_embedding_hnsw"]
    assert "vector_cosine_ops" in (
        definitions["ix_document_chunks_embedding_hnsw"]
    )


def test_documents_owner_id_is_indexed():
    from models import Document

    assert any(
        index.columns.keys() == ["owner_id"]
        for index in Document.__table__.indexes
    )


def test_postgres_app_database_does_not_get_sqlite_options():
    import os
    import subprocess
    import sys
    from pathlib import Path

    # Importing database.py with a PostgreSQL URL must not pass SQLite
    # connect args or register the SQLite PRAGMA listener.
    code = (
        "import database;"
        "print(database.IS_SQLITE);"
        "print(database.engine.dialect.name)"
    )

    env = {
        **os.environ,
        "DATABASE_URL": "postgresql+psycopg://u:p@localhost:1/db",
    }

    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["False", "postgresql"]
