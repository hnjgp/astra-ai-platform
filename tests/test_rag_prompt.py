from rag.context import RAGContext, RAGSource
from rag.prompt import RAGPromptBuilder


def test_rag_prompt_builder_includes_context_and_sources():
    context = RAGContext(
        text=(
            "Python is a programming language.\n\n"
            "FastAPI is a Python web framework."
        ),
        sources=[
            RAGSource(
                document_id=1,
                chunk_index=0,
                score=1.0,
            ),
            RAGSource(
                document_id=1,
                chunk_index=1,
                score=0.8,
            ),
        ],
    )

    builder = RAGPromptBuilder()

    prompt = builder.build(context)

    assert "Python is a programming language." in prompt
    assert "FastAPI is a Python web framework." in prompt
    assert "[Source 1]" in prompt
    assert "[Source 2]" in prompt


def test_rag_prompt_builder_handles_empty_context():
    context = RAGContext(
        text="",
        sources=[],
    )

    builder = RAGPromptBuilder()

    prompt = builder.build(context)

    assert "No relevant information was found" in prompt


def test_rag_prompt_builder_includes_grounding_instructions():
    context = RAGContext(
        text="Python is a programming language.",
        sources=[
            RAGSource(
                document_id=1,
                chunk_index=0,
                score=1.0,
            ),
        ],
    )

    builder = RAGPromptBuilder()

    prompt = builder.build(context)

    assert "provided knowledge base context" in prompt
    assert "Do not invent information" in prompt