from pathlib import Path

from rag.chunking.chunker import DocumentChunker
from rag.ingestion.loader import DocumentLoader
from rag.pipeline import RAGPipeline


def test_pdf_can_be_loaded_and_chunked():

    file_path = Path(
        "documents/sample.pdf"
    )

    loader = DocumentLoader()

    text = loader.load(file_path)

    assert isinstance(text, str)
    assert text.strip()

    chunker = DocumentChunker(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = chunker.split(text)

    assert chunks
    assert all(
        isinstance(chunk, str)
        for chunk in chunks
    )
    assert all(
        chunk.strip()
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
        )
    )

    chunks = pipeline.process(
        file_path
    )

    assert chunks
    assert all(
        isinstance(chunk, str)
        for chunk in chunks
    )
    assert all(
        chunk.strip()
        for chunk in chunks
    )