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


def test_retrieval_service_reranks_real_vector_results(
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

    service = RetrievalService(
        db=rag_db_session,
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

    assert results[0]["content"] == (
        "FastAPI is a Python web framework "
        "for building APIs."
    )