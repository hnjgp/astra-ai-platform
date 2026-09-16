from rag.models import DocumentChunk
from rag.retrieval.retriever import VectorRetriever
from rag.evaluation.retrieval import recall_at_k


def _vector(
    first_value: float,
    second_value: float = 0.0,
) -> list[float]:
    vector = [0.0] * 1536
    vector[0] = first_value
    vector[1] = second_value
    return vector


def test_retrieval_recall_at_k_with_real_vector_database(
    rag_db_session,
):
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

    retriever = VectorRetriever(
        rag_db_session
    )

    results = retriever.search(
        query_embedding=query_embedding,
        top_k=2,
    )

    recall = recall_at_k(
        results=results,
        expected_chunks={(1, 0)},
        k=2,
    )

    assert recall == 1.0