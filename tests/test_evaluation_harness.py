import json

import pytest

from evals.run import (
    check_thresholds,
    document_ids,
    evaluate_agent,
    evaluate_retrieval,
    index_dataset,
    load_dataset,
)
from rag.database import RAGSessionLocal
from rag.embeddings.hashing import HashingEmbedder
from rag.evaluation.answers import (
    groundedness,
    has_citation,
    keyword_correctness,
)
from rag.evaluation.retrieval import mean_reciprocal_rank, reciprocal_rank
from tests.fakes import ScriptedLLMClient


def result(document_id, chunk_index=0):
    return {"document_id": document_id, "chunk_index": chunk_index}


def test_reciprocal_rank_uses_first_relevant_position():
    results = [result(1), result(2), result(3)]

    assert reciprocal_rank(results, {(2, 0)}) == 0.5
    assert reciprocal_rank(results, {(9, 0)}) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank(
        [
            ([result(1)], {(1, 0)}),
            ([result(2), result(1)], {(1, 0)}),
        ]
    ) == pytest.approx(0.75)


def test_keyword_correctness():
    assert keyword_correctness("You get 30 days.", ["30", "days"]) == 1.0
    assert keyword_correctness("You get 20 days.", ["30", "days"]) == 0.5


def test_groundedness_measures_support_in_context():
    context = "Employees receive 30 vacation days per year."

    assert groundedness(
        "Employees receive 30 vacation days [Source 1].",
        context,
    ) == 1.0

    assert groundedness(
        "Employees receive unlimited paid sabbaticals.",
        context,
    ) < 0.5


def test_citation_detection():
    assert has_citation("It is 30 days [Source 1].")
    assert not has_citation("It is 30 days.")


def test_check_thresholds_reports_failures():
    report = {"retrieval": {"recall_at_k": 0.5, "mrr": 0.9}}

    failures = check_thresholds(
        report,
        {
            "retrieval.recall_at_k": 0.8,
            "retrieval.mrr": 0.6,
            "agent.correctness": 0.9,
        },
    )

    assert failures == ["retrieval.recall_at_k = 0.50 is below 0.80"]


def test_offline_retrieval_evaluation(rag_db_session):
    dataset = load_dataset()
    embedder = HashingEmbedder()

    index_dataset(dataset, embedder, RAGSessionLocal)

    report = evaluate_retrieval(
        dataset,
        embedder,
        RAGSessionLocal,
        k=3,
    )

    rag_cases = [
        case
        for case in dataset["cases"]
        if case["expected_document"] is not None
    ]

    assert len(report["cases"]) == len(rag_cases)
    # Baseline for this small dataset; a drop is a regression.
    assert report["recall_at_k"] >= 0.8
    assert report["mrr"] >= 0.6


def test_agent_evaluation_scores_route_answer_and_grounding(
    rag_db_session,
):
    dataset = {
        "documents": [
            {
                "key": "vacation",
                "text": "Every employee receives 30 vacation days per year.",
            }
        ],
        "cases": [
            {
                "question": "How many vacation days?",
                "expected_document": "vacation",
                "expected_keywords": ["30"],
                "expected_route": "rag",
            },
            {
                "question": "What is 12 times 3?",
                "expected_document": None,
                "expected_keywords": ["36"],
                "expected_route": "normal",
            },
        ],
    }

    embedder = HashingEmbedder()
    index_dataset(dataset, embedder, RAGSessionLocal)

    llm = ScriptedLLMClient(
        [
            [
                (
                    "search_knowledge_base",
                    json.dumps({"query": "vacation days per year"}),
                )
            ],
            "Every employee receives 30 vacation days [Source 1].",
            "12 times 3 is 36.",
        ]
    )

    report = evaluate_agent(
        dataset,
        llm,
        embedder,
        RAGSessionLocal,
    )

    assert report["route_accuracy"] == 1.0
    assert report["correctness"] == 1.0
    assert report["groundedness"] == 1.0
    assert report["citation_rate"] == 1.0
    assert document_ids(dataset)["vacation"] >= 9_000_000
