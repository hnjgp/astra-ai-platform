from unittest.mock import Mock

from rag.context import RAGContextBuilder
from rag.embeddings.embedder import Embedder
from rag.models import DocumentChunk
from rag.prompt import RAGPromptBuilder
from rag.retrieval.reranker import KeywordReranker
from rag.retrieval.service import RetrievalService
from rag.service import RAGService


def _vector(
    first_value: float,
    second_value: float = 0.0,
) -> list[float]:
    vector = [0.0] * 1536
    vector[0] = first_value
    vector[1] = second_value
    return vector


def test_rag_service_uses_reranker_with_real_retrieval(
    rag_db_session,
):
    chunks = [
        DocumentChunk(
            document_id=1,
            chunk_index=0,
            content=(
                "Python is a programming language."
            ),
            embedding=_vector(1.0, 0.0),
        ),
        DocumentChunk(
            document_id=1,
            chunk_index=1,
            content=(
                "FastAPI is a Python web framework "
                "for building APIs."
            ),
            embedding=_vector(0.99, 0.01),
        ),
    ]

    rag_db_session.add_all(chunks)
    rag_db_session.commit()

    question = (
        "Which Python web framework "
        "should I use for an API?"
    )

    embedder = Mock()
    embedder.embed.return_value = _vector(
        1.0,
        0.0,
    )

    retrieval_service = RetrievalService(
        db=rag_db_session,
        reranker=KeywordReranker(),
    )

    context_builder = RAGContextBuilder()
    prompt_builder = RAGPromptBuilder()

    service = RAGService(
        embedder=embedder,
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
    )

    prompt = service.build_prompt(
        question=question,
        top_k=2,
    )

    assert "FastAPI" in prompt
    assert (
        "Python is a programming language."
        not in prompt
        or prompt.index("FastAPI")
        < prompt.index(
            "Python is a programming language."
        )
    )