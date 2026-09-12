from sqlalchemy import create_engine
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