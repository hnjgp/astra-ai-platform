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


def test_vector_retriever_filters_by_document_id(
    rag_db_session,
):
    chunks = [
        DocumentChunk(
            document_id=1,
            chunk_index=0,
            content="Python document chunk.",
            embedding=_vector(1.0, 0.0),
        ),
        DocumentChunk(
            document_id=2,
            chunk_index=0,
            content="Different document chunk.",
            embedding=_vector(0.99, 0.01),
        ),
        DocumentChunk(
            document_id=2,
            chunk_index=1,
            content="Another document chunk.",
            embedding=_vector(0.8, 0.6),
        ),
    ]

    rag_db_session.add_all(chunks)
    rag_db_session.commit()

    retriever = VectorRetriever(rag_db_session)

    results = retriever.search(
        query_embedding=_vector(1.0, 0.0),
        top_k=5,
        document_id=2,
    )

    assert len(results) == 2

    assert all(
        result["document_id"] == 2
        for result in results
    )

    assert results[0]["content"] == (
        "Different document chunk."
    )

    assert results[1]["content"] == (
        "Another document chunk."
    )


def test_vector_retriever_returns_empty_when_document_has_no_chunks(
    rag_db_session,
):
    chunk = DocumentChunk(
        document_id=1,
        chunk_index=0,
        content="Existing document chunk.",
        embedding=_vector(1.0, 0.0),
    )

    rag_db_session.add(chunk)
    rag_db_session.commit()

    retriever = VectorRetriever(rag_db_session)

    results = retriever.search(
        query_embedding=_vector(1.0, 0.0),
        top_k=5,
        document_id=999,
    )

    assert results == []


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
def test_vector_retriever_filters_by_score_threshold(
    rag_db_session,
):
    chunks = [
        DocumentChunk(
            document_id=1,
            chunk_index=0,
            content="Highly relevant chunk.",
            embedding=_vector(1.0, 0.0),
        ),
        DocumentChunk(
            document_id=1,
            chunk_index=1,
            content="Moderately relevant chunk.",
            embedding=_vector(0.8, 0.6),
        ),
        DocumentChunk(
            document_id=1,
            chunk_index=2,
            content="Irrelevant chunk.",
            embedding=_vector(0.0, 1.0),
        ),
    ]

    rag_db_session.add_all(chunks)
    rag_db_session.commit()

    retriever = VectorRetriever(rag_db_session)

    results = retriever.search(
        query_embedding=_vector(1.0, 0.0),
        top_k=5,
        score_threshold=0.9,
    )

    assert len(results) == 1

    assert results[0]["content"] == (
        "Highly relevant chunk."
    )

    assert results[0]["score"] >= 0.9
def test_vector_retriever_rejects_invalid_score_threshold(
    rag_db_session,
):
    retriever = VectorRetriever(rag_db_session)

    query_embedding = _vector(1.0, 0.0)

    for invalid_threshold in (-0.1, 1.1):
        try:
            retriever.search(
                query_embedding=query_embedding,
                top_k=5,
                score_threshold=invalid_threshold,
            )
        except ValueError as exc:
            assert str(exc) == (
                "score_threshold must be between zero and one"
            )
        else:
            raise AssertionError(
                "ValueError was not raised"
            )