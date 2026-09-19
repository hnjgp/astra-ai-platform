from unittest.mock import Mock

import pytest

from rag.exceptions import RAGError
from rag.service import RAGService


def make_service(
    embedder=None,
    retrieval_service=None,
    context_builder=None,
    prompt_builder=None,
):
    return RAGService(
        embedder=embedder or Mock(),
        retrieval_service=(
            retrieval_service or Mock()
        ),
        context_builder=(
            context_builder or Mock()
        ),
        prompt_builder=(
            prompt_builder or Mock()
        ),
    )


def test_empty_question_is_rejected():

    service = make_service()

    with pytest.raises(
        ValueError,
        match="question must not be empty",
    ):
        service.build_prompt("   ")


def test_question_is_embedded_before_retrieval():

    embedder = Mock()
    embedder.embed.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    retrieval_service = Mock()
    retrieval_service.search.return_value = []

    context_builder = Mock()
    context_builder.build.return_value = ""

    prompt_builder = Mock()
    prompt_builder.build.return_value = (
        "Context"
    )

    service = make_service(
        embedder=embedder,
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
    )

    service.build_prompt(
        question="What is Astra?"
    )

    embedder.embed.assert_called_once_with(
        "What is Astra?"
    )

    retrieval_service.search.assert_called_once_with(
        query_embedding=[
            0.1,
            0.2,
            0.3,
        ],
        top_k=5,
        document_id=None,
        score_threshold=None,
        query="What is Astra?",
    )


def test_embedding_error_becomes_rag_error():

    embedder = Mock()

    embedder.embed.side_effect = RuntimeError(
        "embedding service unavailable"
    )

    service = make_service(
        embedder=embedder,
    )

    with pytest.raises(
        RAGError,
        match="RAG processing failed",
    ):
        service.build_prompt(
            question="What is Astra?"
        )


def test_retrieval_error_becomes_rag_error():

    embedder = Mock()
    embedder.embed.return_value = [
        0.1,
        0.2,
    ]

    retrieval_service = Mock()

    retrieval_service.search.side_effect = (
        RuntimeError(
            "database unavailable"
        )
    )

    service = make_service(
        embedder=embedder,
        retrieval_service=retrieval_service,
    )

    with pytest.raises(
        RAGError,
        match="RAG processing failed",
    ):
        service.build_prompt(
            question="What is Astra?"
        )


def test_context_error_becomes_rag_error():

    embedder = Mock()
    embedder.embed.return_value = [
        0.1,
        0.2,
    ]

    retrieval_service = Mock()
    retrieval_service.search.return_value = []

    context_builder = Mock()

    context_builder.build.side_effect = (
        RuntimeError(
            "context building failed"
        )
    )

    service = make_service(
        embedder=embedder,
        retrieval_service=retrieval_service,
        context_builder=context_builder,
    )

    with pytest.raises(
        RAGError,
        match="RAG processing failed",
    ):
        service.build_prompt(
            question="What is Astra?"
        )


def test_prompt_error_becomes_rag_error():

    embedder = Mock()
    embedder.embed.return_value = [
        0.1,
        0.2,
    ]

    retrieval_service = Mock()
    retrieval_service.search.return_value = []

    context_builder = Mock()
    context_builder.build.return_value = ""

    prompt_builder = Mock()

    prompt_builder.build.side_effect = (
        RuntimeError(
            "prompt building failed"
        )
    )

    service = make_service(
        embedder=embedder,
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
    )

    with pytest.raises(
        RAGError,
        match="RAG processing failed",
    ):
        service.build_prompt(
            question="What is Astra?"
        )


def test_existing_rag_error_is_preserved():

    embedder = Mock()
    embedder.embed.side_effect = RAGError(
        "known RAG failure"
    )

    service = make_service(
        embedder=embedder,
    )

    with pytest.raises(
        RAGError,
        match="known RAG failure",
    ):
        service.build_prompt(
            question="What is Astra?"
        )