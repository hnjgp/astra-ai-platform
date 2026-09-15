from unittest.mock import Mock

from rag.retrieval.service import RetrievalService


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