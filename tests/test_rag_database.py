import uuid

from sqlalchemy import text

from rag.database import RAGSessionLocal, rag_engine
from rag.models import DocumentChunk


def test_pgvector_extension_is_available():
    with rag_engine.connect() as connection:
        result = connection.execute(
            text(
                """
                SELECT extversion
                FROM pg_extension
                WHERE extname = 'vector'
                """
            )
        ).scalar_one_or_none()

    assert result is not None


def test_document_chunk_can_store_1536_dimension_embedding():
    content = f"Embedding database test {uuid.uuid4()}"

    session = RAGSessionLocal()

    try:
        embedding = [0.1] * 1536

        row = DocumentChunk(
            document_id=999999,
            chunk_index=0,
            content=content,
            embedding=embedding,
        )

        session.add(row)
        session.commit()
        session.refresh(row)

        assert row.id is not None
        assert row.embedding is not None
        assert len(row.embedding) == 1536

        dimensions = session.execute(
            text(
                """
                SELECT vector_dims(embedding)
                FROM document_chunks
                WHERE id = :id
                """
            ),
            {"id": row.id},
        ).scalar_one()

        assert dimensions == 1536

    finally:
        session.close()