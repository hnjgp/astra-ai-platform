import pytest

from rag.evaluation.retrieval import (
    average_recall,
    evaluate_recall_dataset,
    recall_at_k,
)


def test_recall_at_k_returns_one_when_expected_chunk_is_retrieved():
    results = [
        {
            "document_id": 1,
            "chunk_index": 0,
            "content": "Relevant chunk.",
            "score": 0.99,
        }
    ]

    score = recall_at_k(
        results=results,
        expected_chunks={(1, 0)},
        k=1,
    )

    assert score == 1.0


def test_recall_at_k_returns_zero_when_expected_chunk_is_missing():
    results = [
        {
            "document_id": 1,
            "chunk_index": 1,
            "content": "Wrong chunk.",
            "score": 0.99,
        }
    ]

    score = recall_at_k(
        results=results,
        expected_chunks={(1, 0)},
        k=1,
    )

    assert score == 0.0


def test_recall_at_k_returns_zero_when_expected_chunk_is_outside_k():
    results = [
        {
            "document_id": 1,
            "chunk_index": 1,
            "content": "Wrong chunk.",
            "score": 0.99,
        },
        {
            "document_id": 1,
            "chunk_index": 0,
            "content": "Relevant chunk.",
            "score": 0.80,
        },
    ]

    score = recall_at_k(
        results=results,
        expected_chunks={(1, 0)},
        k=1,
    )

    assert score == 0.0


def test_recall_at_k_returns_one_for_multiple_expected_chunks():
    results = [
        {
            "document_id": 1,
            "chunk_index": 0,
            "content": "Relevant chunk one.",
            "score": 0.99,
        },
        {
            "document_id": 1,
            "chunk_index": 1,
            "content": "Relevant chunk two.",
            "score": 0.90,
        },
    ]

    score = recall_at_k(
        results=results,
        expected_chunks={(1, 0), (1, 1)},
        k=2,
    )

    assert score == 1.0


def test_recall_at_k_returns_partial_recall():
    results = [
        {
            "document_id": 1,
            "chunk_index": 0,
            "content": "Relevant chunk.",
            "score": 0.99,
        }
    ]

    score = recall_at_k(
        results=results,
        expected_chunks={(1, 0), (1, 1)},
        k=1,
    )

    assert score == 0.5


def test_recall_at_k_rejects_invalid_k():
    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        recall_at_k(
            results=[],
            expected_chunks={(1, 0)},
            k=0,
        )


def test_recall_at_k_rejects_empty_expected_chunks():
    with pytest.raises(
        ValueError,
        match="expected_chunks must not be empty",
    ):
        recall_at_k(
            results=[],
            expected_chunks=set(),
            k=1,
        )


def test_average_recall_returns_mean():
    score = average_recall(
        [1.0, 0.0, 1.0]
    )

    assert score == pytest.approx(
        2 / 3
    )


def test_average_recall_rejects_empty_evaluations():
    with pytest.raises(
        ValueError,
        match="evaluations must not be empty",
    ):
        average_recall([])


def test_evaluate_recall_dataset_returns_average():
    evaluations = [
        (
            [
                {
                    "document_id": 1,
                    "chunk_index": 0,
                    "content": "First relevant chunk.",
                    "score": 0.99,
                }
            ],
            {(1, 0)},
        ),
        (
            [
                {
                    "document_id": 1,
                    "chunk_index": 1,
                    "content": "Wrong chunk.",
                    "score": 0.90,
                }
            ],
            {(1, 0)},
        ),
        (
            [
                {
                    "document_id": 1,
                    "chunk_index": 2,
                    "content": "Third relevant chunk.",
                    "score": 0.80,
                }
            ],
            {(1, 2)},
        ),
    ]

    score = evaluate_recall_dataset(
        evaluations=evaluations,
        k=1,
    )

    assert score == pytest.approx(
        2 / 3
    )


def test_evaluate_recall_dataset_rejects_empty_evaluations():
    with pytest.raises(
        ValueError,
        match="evaluations must not be empty",
    ):
        evaluate_recall_dataset(
            evaluations=[],
            k=1,
        )