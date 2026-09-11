from pathlib import Path

from rag.chunking.chunker import DocumentChunker
from rag.ingestion.loader import DocumentLoader


class RAGPipeline:
    """
    Load a document and split its text into chunks.
    """

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:

        self.loader = (
            loader
            if loader is not None
            else DocumentLoader()
        )

        self.chunker = (
            chunker
            if chunker is not None
            else DocumentChunker()
        )

    def process(
        self,
        file_path: str | Path,
    ) -> list[str]:
        """
        Load a document and return its chunks.
        """

        text = self.loader.load(
            file_path
        )

        return self.chunker.split(
            text
        )