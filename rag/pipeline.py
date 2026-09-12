from pathlib import Path

from rag.chunking.chunker import DocumentChunker
from rag.embeddings.embedder import Embedder
from rag.ingestion.loader import DocumentLoader
from rag.models import EmbeddedChunk


class RAGPipeline:
    def __init__(
        self,
        loader: DocumentLoader | None = None,
        chunker: DocumentChunker | None = None,
        embedder: Embedder | None = None,
    ):
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or DocumentChunker()
        self.embedder = embedder

    def process(self, path: Path) -> list[EmbeddedChunk]:
        text = self.loader.load(path)

        chunks = self.chunker.split(text)

        if self.embedder is None:
            raise ValueError("embedder is required")

        return [
            EmbeddedChunk(
                text=chunk,
                embedding=self.embedder.embed(chunk),
            )
            for chunk in chunks
        ]