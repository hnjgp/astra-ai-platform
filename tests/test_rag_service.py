from unittest.mock import Mock

from rag.context import RAGContext, RAGSource
from rag.service import RAGService


def test_rag_service_builds_prompt_from_question():
    embedder = Mock()
    retrieval_service = Mock()
    context_builder = Mock()
    prompt_builder = Mock()

    query_embedding = [0.1, 0.2, 0.3]

    embedder.embed.return_value = query_embedding

    retrieval_results = [
        {
            "document_id": 1,
            "chunk_index": 0,
            "content": "Python is a programming language.",
            "score": 0.95,
        }
    ]

    retrieval_service.search.return_value = retrieval_results

    context = RAGContext(
        text="Python is a programming language.",
        sources=[
            RAGSource(
                document_id=1,
                chunk_index=0,
                score=0.95,
            )
        ],
    )

    context_builder.build.return_value = context
    prompt_builder.build.return_value = (
        "Use the knowledge base to answer the question."
    )

    service = RAGService(
        embedder=embedder,
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
    )

    prompt = service.build_prompt(
        question="What is Python?"
    )

    assert prompt == (
        "Use the knowledge base to answer the question."
    )

    embedder.embed.assert_called_once_with(
        "What is Python?"
    )

    retrieval_service.search.assert_called_once_with(
        query_embedding=query_embedding,
        top_k=5,
        document_id=None,
        score_threshold=None,
    )

    context_builder.build.assert_called_once_with(
        retrieval_results
    )

    prompt_builder.build.assert_called_once_with(
        context
    )


def test_rag_service_passes_retrieval_options():
    embedder = Mock()
    retrieval_service = Mock()
    context_builder = Mock()
    prompt_builder = Mock()

    embedder.embed.return_value = [0.1, 0.2]

    retrieval_service.search.return_value = []

    context = RAGContext(
        text="",
        sources=[],
    )

    context_builder.build.return_value = context

    prompt_builder.build.return_value = (
        "No relevant information was found."
    )

    service = RAGService(
        embedder=embedder,
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
    )

    service.build_prompt(
        question="What is Python?",
        top_k=3,
        document_id=10,
        score_threshold=0.8,
    )

    retrieval_service.search.assert_called_once_with(
        query_embedding=[0.1, 0.2],
        top_k=3,
        document_id=10,
        score_threshold=0.8,
    )


def test_rag_service_builds_prompt_when_no_context_is_found():
    embedder = Mock()
    retrieval_service = Mock()
    context_builder = Mock()
    prompt_builder = Mock()

    embedder.embed.return_value = [0.1, 0.2]

    retrieval_service.search.return_value = []

    context = RAGContext(
        text="",
        sources=[],
    )

    context_builder.build.return_value = context

    prompt_builder.build.return_value = (
        "No relevant information was found in the knowledge base."
    )

    service = RAGService(
        embedder=embedder,
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
    )

    prompt = service.build_prompt(
        question="What is the vacation policy?"
    )

    assert prompt == (
        "No relevant information was found in the knowledge base."
    )

    retrieval_service.search.assert_called_once_with(
        query_embedding=[0.1, 0.2],
        top_k=5,
        document_id=None,
        score_threshold=None,
    )

    context_builder.build.assert_called_once_with(
        retrieval_service.search.return_value
    )

    prompt_builder.build.assert_called_once_with(
        context
    )