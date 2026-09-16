from unittest.mock import Mock

from rag.models import DocumentChunk
from rag.retrieval.reranker import KeywordReranker
from rag.retrieval.service import RetrievalService


def _vector(
    first_value: float,
    second_value: float = 0.0,
) -> list[float]:
    vector = [0.0] * 1536
    vector[0] = first_value
    vector[1] = second_value
    return vector


def test_retrieval_service_forwards_search_request():
    retriever = Mock()

    expected_results = [
        {
            "document_id": 1,
            "chunk_index": 0,
            "content": "Astra is an AI platform.",
            "score": 0.95,
        }
    ]

    retriever.search.return_value = expected_results

    service = RetrievalService(
        db=Mock(),
        retriever=retriever,
    )

    results = service.search(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=5,
        document_id=1,
        score_threshold=0.8,
    )

    assert results == expected_results

    retriever.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=5,
        document_id=1,
        score_threshold=0.8,
    )


def test_retrieval_service_creates_retriever_when_not_provided():
    service = RetrievalService(db=Mock())

    assert service.retriever is not None


def test_retrieval_service_uses_default_search_options():
    retriever = Mock()

    expected_results = [
        {
            "document_id": 1,
            "chunk_index": 0,
            "content": "Astra is an AI platform.",
            "score": 0.95,
        }
    ]

    retriever.search.return_value = expected_results

    service = RetrievalService(
        db=Mock(),
        retriever=retriever,
    )

    results = service.search(
        query_embedding=[0.1, 0.2, 0.3],
    )

    assert results == expected_results

    retriever.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=5,
        document_id=None,
        score_threshold=None,
    )


def test_retrieval_service_uses_reranker(
    rag_db_session,
):
    retriever = Mock()

    retriever.search.return_value = [
        {
            "document_id": 1,
            "chunk_index": 0,
            "content": (
                "Python is a programming language."
            ),
            "score": 0.99,
        },
        {
            "document_id": 1,
            "chunk_index": 1,
            "content": (
                "FastAPI is a Python web framework "
                "for building APIs."
            ),
            "score": 0.98,
        },
    ]

    service = RetrievalService(
        db=rag_db_session,
        retriever=retriever,
        reranker=KeywordReranker(),
    )

    results = service.search(
        query_embedding=_vector(1.0, 0.0),
        top_k=2,
        query=(
            "Which Python web framework "
            "should I use for an API?"
        ),
    )

    assert len(results) == 2
    assert results[0]["chunk_index"] == 1
    assert results[1]["chunk_index"] == 0


def test_retrieval_service_works_without_reranker():
    retriever = Mock()

    expected_results = [
        {
            "document_id": 1,
            "chunk_index": 0,
            "content": "Python is a programming language.",
            "score": 0.99,
        }
    ]

    retriever.search.return_value = expected_results

    service = RetrievalService(
        db=Mock(),
        retriever=retriever,
    )

    results = service.search(
        query_embedding=_vector(1.0, 0.0),
        top_k=1,
    )

    assert results == expected_results

    retriever.search.assert_called_once_with(
        query_embedding=_vector(1.0, 0.0),
        top_k=1,
        document_id=None,
        score_threshold=None,
    )


def test_retrieval_service_requires_query_when_reranker_enabled():
    retriever = Mock()

    retriever.search.return_value = []

    service = RetrievalService(
        db=Mock(),
        retriever=retriever,
        reranker=KeywordReranker(),
    )

    try:
        service.search(
            query_embedding=_vector(1.0, 0.0),
        )
    except ValueError as exc:
        assert str(exc) == (
            "query is required when reranker is enabled"
        )
    else:
        raise AssertionError(
            "ValueError was not raised"
        )