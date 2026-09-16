from rag.evaluation.retrieval import recall_at_k
from rag.models import DocumentChunk
from rag.retrieval.retriever import VectorRetriever


def _vector(
    first_value: float,
    second_value: float = 0.0,
) -> list[float]:
    vector = [0.0] * 1536
    vector[0] = first_value
    vector[1] = second_value
    return vector


def test_retrieval_baseline(
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
            embedding=_vector(0.0, 1.0),
        ),
        DocumentChunk(
            document_id=1,
            chunk_index=3,
            content="Docker packages applications into containers.",
            embedding=_vector(-0.8, 0.6),
        ),
        DocumentChunk(
            document_id=1,
            chunk_index=4,
            content="Embeddings represent text as vectors.",
            embedding=_vector(-1.0, 0.0),
        ),
    ]

    rag_db_session.add_all(chunks)
    rag_db_session.commit()

    retriever = VectorRetriever(
        rag_db_session
    )

    questions = [
        {
            "query_embedding": _vector(0.95, 0.05),
            "expected_chunks": {(1, 0)},
        },
        {
            "query_embedding": _vector(0.75, 0.66),
            "expected_chunks": {(1, 1)},
        },
        {
            "query_embedding": _vector(0.05, 0.95),
            "expected_chunks": {(1, 2)},
        },
    ]

    scores = {
        1: [],
        3: [],
        5: [],
    }

    for question in questions:
        results = retriever.search(
            query_embedding=question["query_embedding"],
            top_k=5,
        )

        for k in scores:
            score = recall_at_k(
                results=results,
                expected_chunks=question["expected_chunks"],
                k=k,
            )

            scores[k].append(score)

    average_scores = {
        k: sum(values) / len(values)
        for k, values in scores.items()
    }

    print(
        "\nRetrieval baseline:"
        f"\nRecall@1: {average_scores[1]:.2f}"
        f"\nRecall@3: {average_scores[3]:.2f}"
        f"\nRecall@5: {average_scores[5]:.2f}"
    )

    assert average_scores[1] == 1.0
    assert average_scores[3] == 1.0
    assert average_scores[5] == 1.0