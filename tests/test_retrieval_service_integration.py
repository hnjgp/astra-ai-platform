from rag.models import DocumentChunk
from rag.retrieval.service import RetrievalService


def _vector(
    first_value: float,
    second_value: float = 0.0,
) -> list[float]:
    vector = [0.0] * 1536
    vector[0] = first_value
    vector[1] = second_value
    return vector


def test_retrieval_service_searches_real_database(
    rag_db_session,
):
    chunks = [
        DocumentChunk(
            document_id=1,
            chunk_index=0,
            content="Python is a programming language.",
            embedding=_vector(1.0, 0.0),
        ),
        DocumentChunk(
            document_id=1,
            chunk_index=1,
            content="FastAPI is a web framework.",
            embedding=_vector(0.8, 0.6),
        ),
        DocumentChunk(
            document_id=1,
            chunk_index=2,
            content="PostgreSQL is a database.",
            embedding=_vector(-1.0, 0.0),
        ),
    ]

    rag_db_session.add_all(chunks)
    rag_db_session.commit()

    service = RetrievalService(rag_db_session)

    results = service.search(
        query_embedding=_vector(1.0, 0.0),
        top_k=2,
    )

    assert len(results) == 2

    assert results[0]["content"] == (
        "Python is a programming language."
    )

    assert results[1]["content"] == (
        "FastAPI is a web framework."
    )

    assert results[0]["score"] > results[1]["score"]