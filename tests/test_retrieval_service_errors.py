from unittest.mock import Mock

import pytest

from rag.retrieval.service import RetrievalService


def test_score_threshold_is_passed_to_retriever():

    retriever = Mock()

    retriever.search.return_value = []

    service = RetrievalService(
        db=Mock(),
        retriever=retriever,
    )

    service.search(
        query_embedding=[
            0.1,
            0.2,
        ],
        top_k=3,
        document_id=10,
        score_threshold=0.8,
    )

    retriever.search.assert_called_once_with(
        query_embedding=[
            0.1,
            0.2,
        ],
        top_k=3,
        document_id=10,
        score_threshold=0.8,
    )


def test_reranker_requires_query():

    retriever = Mock()

    retriever.search.return_value = []

    reranker = Mock()

    service = RetrievalService(
        db=Mock(),
        retriever=retriever,
        reranker=reranker,
    )

    with pytest.raises(
        ValueError,
        match="query is required when reranker is enabled",
    ):
        service.search(
            query_embedding=[
                0.1,
                0.2,
            ],
        )

    reranker.rerank.assert_not_called()