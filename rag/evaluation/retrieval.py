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

def reciprocal_rank(
    results: list[dict],
    expected_chunks: set[tuple[int | None, int]],
) -> float:
    """
    1 / position of the first relevant result (0 when none is found).
    """

    if not expected_chunks:
        raise ValueError(
            "expected_chunks must not be empty"
        )

    for position, result in enumerate(results, start=1):
        if (
            result["document_id"],
            result["chunk_index"],
        ) in expected_chunks:
            return 1 / position

    return 0.0


def mean_reciprocal_rank(
    evaluations: list[
        tuple[
            list[dict],
            set[tuple[int | None, int]],
        ]
    ],
) -> float:
    if not evaluations:
        raise ValueError(
            "evaluations must not be empty"
        )

    return average_recall(
        [
            reciprocal_rank(results, expected_chunks)
            for results, expected_chunks in evaluations
        ]
    )
