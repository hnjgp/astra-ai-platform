from pathlib import Path

import pytest

from rag.pipeline import RAGPipeline


class FakeEmbedder:
    def __init__(self):
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)

        return [0.1, 0.2, 0.3]


def test_rag_pipeline_creates_embeddings_for_chunks(
    tmp_path: Path,
):
    document = tmp_path / "document.txt"

    document.write_text(
        "Python is great. " * 100,
        encoding="utf-8",
    )

    embedder = FakeEmbedder()

    pipeline = RAGPipeline(
        embedder=embedder,
    )

    results = pipeline.process(document)

    assert results

    assert all(result.text for result in results)

    assert all(
        result.embedding == [0.1, 0.2, 0.3]
        for result in results
    )

    assert len(embedder.calls) == len(results)


def test_rag_pipeline_requires_embedder(
    tmp_path: Path,
):
    document = tmp_path / "document.txt"

    document.write_text(
        "Python is great.",
        encoding="utf-8",
    )

    pipeline = RAGPipeline()

    with pytest.raises(
        ValueError,
        match="embedder is required",
    ):
        pipeline.process(document)