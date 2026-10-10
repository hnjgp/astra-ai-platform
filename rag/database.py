from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config import settings


rag_engine = create_engine(
    settings.RAG_DATABASE_URL,
    pool_pre_ping=True,
)

RAGSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=rag_engine,
)


def get_rag_db():
    db = RAGSessionLocal()

    try:
        yield db
    finally:
        db.close()

def init_rag_database() -> None:
    """
    Create the pgvector extension, the chunk table and its indexes.

    Idempotent: safe to run on every start, also on a database whose
    table was created before the indexes existed.
    """

    from rag.models import DocumentChunk, RAGBase

    with rag_engine.begin() as connection:
        connection.execute(
            text("CREATE EXTENSION IF NOT EXISTS vector")
        )

        RAGBase.metadata.create_all(bind=connection)

        for index in DocumentChunk.__table__.indexes:
            index.create(
                bind=connection,
                checkfirst=True,
            )


def check_rag_database() -> None:
    with rag_engine.connect() as connection:
        connection.execute(text("SELECT 1"))
