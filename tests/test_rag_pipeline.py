from pathlib import Path

from rag.chunking.chunker import DocumentChunker
from rag.pipeline import RAGPipeline


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


def test_pdf_can_be_loaded_and_chunked():

    file_path = Path(
        "documents/sample.pdf"
    )

    pipeline = RAGPipeline(
        chunker=DocumentChunker(
            chunk_size=500,
            chunk_overlap=50,
        ),
        embedder=FakeEmbedder(),
    )

    chunks = pipeline.process(
        file_path
    )

    assert chunks

    assert all(
        chunk.text
        for chunk in chunks
    )

    assert all(
        chunk.embedding == [0.1, 0.2, 0.3]
        for chunk in chunks
    )


def test_rag_pipeline_processes_document():

    file_path = Path(
        "documents/sample.pdf"
    )

    pipeline = RAGPipeline(
        chunker=DocumentChunker(
            chunk_size=500,
            chunk_overlap=50,
        ),
        embedder=FakeEmbedder(),
    )

    chunks = pipeline.process(
        file_path
    )

    assert len(chunks) > 1

    assert all(
        chunk.text
        for chunk in chunks
    )

    assert all(
        isinstance(
            chunk.embedding,
            list
        )
        for chunk in chunks
    )

    assert all(
        len(chunk.embedding) == 3
        for chunk in chunks
    )