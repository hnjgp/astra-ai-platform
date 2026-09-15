import pytest
from pydantic import ValidationError

from rag.context import RAGContext, RAGContextBuilder, RAGSource


def test_rag_context_builder_builds_context_and_sources():
    results = [
        {
            "document_id": 1,
            "chunk_index": 0,
            "content": "Python is a programming language.",
            "score": 0.95,
        },
        {
            "document_id": 1,
            "chunk_index": 1,
            "content": "FastAPI is a web framework.",
            "score": 0.80,
        },
    ]

    builder = RAGContextBuilder()

    context = builder.build(results)

    assert (
        context.text
        == "[Source 1 | Document 1 | Chunk 0 | Score 0.950]\n"
        "Python is a programming language.\n\n"
        "[Source 2 | Document 1 | Chunk 1 | Score 0.800]\n"
        "FastAPI is a web framework."
    )

    assert len(context.sources) == 2

    assert context.sources[0].document_id == 1
    assert context.sources[0].chunk_index == 0
    assert context.sources[0].score == 0.95

    assert context.sources[1].document_id == 1
    assert context.sources[1].chunk_index == 1
    assert context.sources[1].score == 0.80


def test_rag_context_builder_handles_empty_results():
    builder = RAGContextBuilder()

    context = builder.build([])

    assert context.text == ""
    assert context.sources == []


def test_rag_source_rejects_extra_fields():
    with pytest.raises(ValidationError):
        RAGSource(
            document_id=1,
            chunk_index=0,
            score=0.95,
            extra_field="not allowed",
        )


def test_rag_context_rejects_extra_fields():
    with pytest.raises(ValidationError):
        RAGContext(
            text="Some context",
            sources=[],
            extra_field="not allowed",
        )


def test_rag_context_requires_text_and_sources():
    with pytest.raises(ValidationError):
        RAGContext()


def test_rag_source_requires_required_fields():
    with pytest.raises(ValidationError):
        RAGSource()