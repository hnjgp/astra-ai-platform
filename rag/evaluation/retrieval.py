def recall_at_k(
    results: list[dict],
    expected_chunks: set[tuple[int | None, int]],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not expected_chunks:
        raise ValueError(
            "expected_chunks must not be empty"
        )

    top_results = results[:k]

    retrieved_chunks = {
        (
            result["document_id"],
            result["chunk_index"],
        )
        for result in top_results
    }

    relevant_chunks = (
        retrieved_chunks & expected_chunks
    )

    return (
        len(relevant_chunks)
        / len(expected_chunks)
    )


def average_recall(
    evaluations: list[float],
) -> float:
    if not evaluations:
        raise ValueError(
            "evaluations must not be empty"
        )

    return sum(evaluations) / len(evaluations)


def evaluate_recall_dataset(
    evaluations: list[
        tuple[
            list[dict],
            set[tuple[int | None, int]],
        ]
    ],
    k: int,
) -> float:
    if not evaluations:
        raise ValueError(
            "evaluations must not be empty"
        )

    scores = [
        recall_at_k(
            results=results,
            expected_chunks=expected_chunks,
            k=k,
        )
        for results, expected_chunks in evaluations
    ]

    return average_recall(scores)