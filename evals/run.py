"""
Evaluate retrieval (offline or live) and the agent (live only).

    python -m evals.run                 # retrieval, HashingEmbedder
    python -m evals.run --live          # OpenAI embeddings + agent eval

The dataset is indexed into the configured RAG database under reserved
document ids and removed afterwards. Exit code 1 when a metric is below
its threshold, so the command can be used as a regression gate.
"""

import argparse
import json
import sys
from pathlib import Path

from rag.context import RAGContext
from rag.evaluation.answers import (
    groundedness,
    has_citation,
    keyword_correctness,
)
from rag.evaluation.retrieval import (
    average_recall,
    mean_reciprocal_rank,
    recall_at_k,
)
from rag.indexer import DocumentIndexer
from rag.retrieval.service import RetrievalService


DATASET_PATH = Path(__file__).with_name("dataset.json")

# Reserved ids so evaluation chunks never collide with real documents.
EVAL_DOCUMENT_ID_OFFSET = 9_000_000


def load_dataset(path: Path = DATASET_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def document_ids(dataset: dict) -> dict[str, int]:
    return {
        document["key"]: EVAL_DOCUMENT_ID_OFFSET + index
        for index, document in enumerate(dataset["documents"])
    }


def index_dataset(
    dataset: dict,
    embedder,
    session_factory,
) -> DocumentIndexer:

    indexer = DocumentIndexer(
        embedder=embedder,
        session_factory=session_factory,
    )

    ids = document_ids(dataset)

    for document in dataset["documents"]:
        indexer.index(
            document_id=ids[document["key"]],
            text=document["text"],
        )

    return indexer


def evaluate_retrieval(
    dataset: dict,
    embedder,
    session_factory,
    k: int = 3,
) -> dict:

    ids = document_ids(dataset)
    evaluations = []
    cases = []

    with session_factory() as db:
        service = RetrievalService(db)

        for case in dataset["cases"]:
            if case["expected_document"] is None:
                continue

            results = service.search(
                query_embedding=embedder.embed(case["question"]),
                top_k=k,
                document_ids=list(ids.values()),
            )

            expected = {
                (ids[case["expected_document"]], 0)
            }

            evaluations.append((results, expected))

            cases.append(
                {
                    "question": case["question"],
                    "recall_at_k": recall_at_k(results, expected, k),
                    "retrieved": [
                        result["document_id"] - EVAL_DOCUMENT_ID_OFFSET
                        for result in results
                    ],
                }
            )

    return {
        "k": k,
        "recall_at_k": average_recall(
            [case["recall_at_k"] for case in cases]
        ),
        "mrr": mean_reciprocal_rank(evaluations),
        "cases": cases,
    }


def evaluate_agent(
    dataset: dict,
    llm_client,
    embedder,
    session_factory,
) -> dict:
    """
    Run each case through the real agent and score the answer.
    """

    from agents.agent import Agent
    from rag.service import RAGService
    from services.agent_service import AGENT_INSTRUCTIONS
    from tools.executor import ToolExecutor
    from tools.knowledge_base import KnowledgeBaseSearchTool
    from tools.registry import TOOL_REGISTRY

    allowed_ids = list(document_ids(dataset).values())
    cases = []

    for case in dataset["cases"]:
        contexts: list[str] = []

        def retrieve(query: str, top_k: int) -> RAGContext:
            with session_factory() as db:
                context = RAGService(
                    embedder=embedder,
                    retrieval_service=RetrievalService(db),
                ).retrieve(
                    question=query,
                    top_k=top_k,
                    document_ids=allowed_ids,
                )

            contexts.append(context.text)
            return context

        knowledge_base = KnowledgeBaseSearchTool(retrieve=retrieve)

        executor = ToolExecutor(
            registry={
                **TOOL_REGISTRY,
                knowledge_base.name: knowledge_base,
            }
        )

        result = Agent(
            llm_client=llm_client,
            tool_executor=executor.execute,
        ).run_detailed(
            message=case["question"],
            tools=executor.tool_definitions("user"),
            instructions=AGENT_INSTRUCTIONS,
        )

        scored = {
            "question": case["question"],
            "expected_route": case["expected_route"],
            "route": result.route,
            "route_correct": result.route == case["expected_route"],
            "correctness": keyword_correctness(
                result.output_text,
                case["expected_keywords"],
            ),
            "answer": result.output_text,
        }

        if case["expected_route"] == "rag":
            scored["groundedness"] = groundedness(
                result.output_text,
                "\n".join(contexts),
            )
            scored["cited"] = has_citation(result.output_text)

        cases.append(scored)

    rag_cases = [case for case in cases if "groundedness" in case]

    return {
        "route_accuracy": average_recall(
            [float(case["route_correct"]) for case in cases]
        ),
        "correctness": average_recall(
            [case["correctness"] for case in cases]
        ),
        "groundedness": average_recall(
            [case["groundedness"] for case in rag_cases]
        ),
        "citation_rate": average_recall(
            [float(case["cited"]) for case in rag_cases]
        ),
        "cases": cases,
    }


def check_thresholds(
    report: dict,
    thresholds: dict[str, float],
) -> list[str]:
    """
    Return the failed checks as readable messages.
    """

    failures = []

    for path, minimum in thresholds.items():
        section, metric = path.split(".")

        if section not in report:
            continue

        value = report[section][metric]

        if value < minimum:
            failures.append(
                f"{path} = {value:.2f} is below {minimum:.2f}"
            )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min-recall", type=float, default=0.8)
    parser.add_argument("--min-mrr", type=float, default=0.6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    from rag.database import RAGSessionLocal, init_rag_database

    init_rag_database()

    if args.live:
        from llm.client import LLMClient
        from services.providers import get_embedder

        embedder = get_embedder()
    else:
        from rag.embeddings.hashing import HashingEmbedder

        embedder = HashingEmbedder()

    dataset = load_dataset()
    indexer = index_dataset(dataset, embedder, RAGSessionLocal)

    try:
        report = {
            "mode": "live" if args.live else "offline",
            "retrieval": evaluate_retrieval(
                dataset,
                embedder,
                RAGSessionLocal,
                k=args.k,
            ),
        }

        if args.live:
            report["agent"] = evaluate_agent(
                dataset,
                LLMClient(),
                embedder,
                RAGSessionLocal,
            )

    finally:
        for document_id in document_ids(dataset).values():
            indexer.remove(document_id)

    failures = check_thresholds(
        report,
        {
            "retrieval.recall_at_k": args.min_recall,
            "retrieval.mrr": args.min_mrr,
        },
    )

    report["passed"] = not failures
    report["failures"] = failures

    text = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        args.output.write_text(text, encoding="utf-8")

    print(text)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
