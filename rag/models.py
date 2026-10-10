from dataclasses import dataclass

from sqlalchemy import Index, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


@dataclass
class EmbeddedChunk:
    text: str
    embedding: list[float]


class RAGBase(DeclarativeBase):
    pass


class DocumentChunk(RAGBase):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    document_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(1536),
        nullable=False,
    )

    __table_args__ = (
        # Filtering and deleting chunks by document.
        Index(
            "ix_document_chunks_document_id",
            "document_id",
        ),
        # Approximate nearest-neighbour search for cosine distance;
        # without it every query scans all vectors.
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={
                "m": 16,
                "ef_construction": 64,
            },
            postgresql_ops={
                "embedding": "vector_cosine_ops",
            },
        ),
    )