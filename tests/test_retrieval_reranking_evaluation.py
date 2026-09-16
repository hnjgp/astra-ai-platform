from rag.evaluation.retrieval import (
    average_recall,
    recall_at_k,
)
from rag.models import DocumentChunk
from rag.retrieval.reranker import KeywordReranker
from rag.retrieval.retriever import VectorRetriever
from rag.retrieval.service import RetrievalService


def _vector(
    first_value: float,
    second_value: float = 0.0,
) -> list[float]:
    vector = [0.0] * 1536
    vector[0] = first_value
    vector[1] = second_value
    return vector


def test_reranking_improves_recall_across_multiple_questions(
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
        DocumentChunk(
            document_id=1,
            chunk_index=2,
            content=(
                "PostgreSQL is a relational database "
                "for storing application data."
            ),
            embedding=_vector(0.98, 0.02),
        ),
        DocumentChunk(
            document_id=1,
            chunk_index=3,
            content=(
                "Docker packages applications into "
                "portable containers."
            ),
            embedding=_vector(0.97, 0.03),
        ),
    ]

    rag_db_session.add_all(chunks)
    rag_db_session.commit()

    retriever = VectorRetriever(
        rag_db_session
    )

    reranked_service = RetrievalService(
        db=rag_db_session,
        reranker=KeywordReranker(),
    )

    evaluations = [
        {
            "question": (
                "Which Python web framework "
                "should I use for an API?"
            ),
            "query_embedding": _vector(
                1.0,
                0.0,
            ),
            "expected_chunks": {(1, 1)},
        },
        {
            "question": (
                "What programming language "
                "is Python?"
            ),
            "query_embedding": _vector(
                1.0,
                0.0,
            ),
            "expected_chunks": {(1, 0)},
        },
        {
            "question": (
                "Which database should I use "
                "for storing application data?"
            ),
            "query_embedding": _vector(
                0.98,
                0.02,
            ),
            "expected_chunks": {(1, 2)},
        },
    ]

    baseline_recall_at_1 = []
    baseline_recall_at_3 = []
    baseline_recall_at_5 = []

    reranked_recall_at_1 = []
    reranked_recall_at_3 = []
    reranked_recall_at_5 = []

    for evaluation in evaluations:
        baseline_results = retriever.search(
            query_embedding=evaluation[
                "query_embedding"
            ],
            top_k=5,
        )

        reranked_results = (
            reranked_service.search(
                query_embedding=evaluation[
                    "query_embedding"
                ],
                top_k=5,
                query=evaluation["question"],
            )
        )

        expected_chunks = evaluation[
            "expected_chunks"
        ]

        baseline_recall_at_1.append(
            recall_at_k(
                results=baseline_results,
                expected_chunks=expected_chunks,
                k=1,
            )
        )

        baseline_recall_at_3.append(
            recall_at_k(
                results=baseline_results,
                expected_chunks=expected_chunks,
                k=3,
            )
        )

        baseline_recall_at_5.append(
            recall_at_k(
                results=baseline_results,
                expected_chunks=expected_chunks,
                k=5,
            )
        )

        reranked_recall_at_1.append(
            recall_at_k(
                results=reranked_results,
                expected_chunks=expected_chunks,
                k=1,
            )
        )

        reranked_recall_at_3.append(
            recall_at_k(
                results=reranked_results,
                expected_chunks=expected_chunks,
                k=3,
            )
        )

        reranked_recall_at_5.append(
            recall_at_k(
                results=reranked_results,
                expected_chunks=expected_chunks,
                k=5,
            )
        )

    baseline_1 = average_recall(
        baseline_recall_at_1
    )
    baseline_3 = average_recall(
        baseline_recall_at_3
    )
    baseline_5 = average_recall(
        baseline_recall_at_5
    )

    reranked_1 = average_recall(
        reranked_recall_at_1
    )
    reranked_3 = average_recall(
        reranked_recall_at_3
    )
    reranked_5 = average_recall(
        reranked_recall_at_5
    )

    print(
        "\nRetrieval evaluation:"
        f"\nBaseline Recall@1: {baseline_1:.2f}"
        f"\nBaseline Recall@3: {baseline_3:.2f}"
        f"\nBaseline Recall@5: {baseline_5:.2f}"
        f"\nReranked Recall@1: {reranked_1:.2f}"
        f"\nReranked Recall@3: {reranked_3:.2f}"
        f"\nReranked Recall@5: {reranked_5:.2f}"
    )

    assert reranked_1 >= baseline_1
    assert reranked_3 >= baseline_3
    assert reranked_5 >= baseline_5