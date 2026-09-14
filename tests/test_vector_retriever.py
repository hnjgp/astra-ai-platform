import math

from rag.models import DocumentChunk
from rag.retrieval.retriever import VectorRetriever


def _vector(
    first_value: float,
    second_value: float = 0.0,
) -> list[float]:
    vector = [0.0] * 1536
    vector[0] = first_value
    vector[1] = second_value
    return vector


def test_vector_retriever_returns_closest_chunks(rag_db_session):
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

    query_embedding = _vector(1.0, 0.0)

    retriever = VectorRetriever(rag_db_session)

    results = retriever.search(
        query_embedding=query_embedding,
        top_k=2,
    )

    assert len(results) == 2

    assert results[0]["content"] == (
        "Python is a programming language."
    )

    assert results[1]["content"] == (
        "FastAPI is a web framework."
    )

    assert math.isclose(
        results[0]["score"],
        1.0,
        abs_tol=0.001,
    )

    assert math.isclose(
        results[1]["score"],
        0.8,
        abs_tol=0.001,
    )

    assert results[0]["score"] > results[1]["score"]


def test_vector_retriever_rejects_invalid_query_embedding(
    rag_db_session,
):
    retriever = VectorRetriever(rag_db_session)

    try:
        retriever.search(
            query_embedding=[],
            top_k=5,
        )
    except ValueError as exc:
        assert str(exc) == "query_embedding must not be empty"
    else:
        raise AssertionError("ValueError was not raised")


def test_vector_retriever_rejects_invalid_top_k(
    rag_db_session,
):
    retriever = VectorRetriever(rag_db_session)

    query_embedding = _vector(1.0, 0.0)

    try:
        retriever.search(
            query_embedding=query_embedding,
            top_k=0,
        )
    except ValueError as exc:
        assert str(exc) == "top_k must be greater than zero"
    else:
        raise AssertionError("ValueError was not raised")