from sqlalchemy import delete

from rag.chunking.chunker import DocumentChunker
from rag.embeddings.embedder import Embedder
from rag.exceptions import RAGError
from rag.models import DocumentChunk


class DocumentIndexer:
    """
    Writes a document into the vector store: chunk, embed, store.
    """

    def __init__(
        self,
        embedder: Embedder,
        session_factory,
        chunker: DocumentChunker | None = None,
    ):
        self.embedder = embedder
        self.session_factory = session_factory
        self.chunker = chunker or DocumentChunker()

    def index(
        self,
        document_id: int,
        text: str,
    ) -> int:
        """
        Replace the stored chunks of a document. Returns the chunk count.
        """

        chunks = self.chunker.split(text)

        if not chunks:
            raise ValueError(
                "document has no text to index"
            )

        # Embed before opening the transaction so the database is not
        # held while waiting for the embedding provider.
        try:
            embeddings = self.embedder.embed_many(chunks)
        except Exception as exc:
            raise RAGError(
                "embedding provider is unavailable"
            ) from exc

        if len(embeddings) != len(chunks):
            raise ValueError(
                "embedder returned a wrong number of embeddings"
            )

        try:
            self._replace_chunks(
                document_id,
                chunks,
                embeddings,
            )
        except Exception as exc:
            raise RAGError(
                "knowledge base storage failed"
            ) from exc

        return len(chunks)

    def _replace_chunks(
        self,
        document_id: int,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:

        with self.session_factory() as db:
            with db.begin():
                db.execute(
                    delete(DocumentChunk).where(
                        DocumentChunk.document_id == document_id
                    )
                )

                db.add_all(
                    [
                        DocumentChunk(
                            document_id=document_id,
                            chunk_index=index,
                            content=chunk,
                            embedding=embedding,
                        )
                        for index, (chunk, embedding) in enumerate(
                            zip(chunks, embeddings)
                        )
                    ]
                )

    def remove(
        self,
        document_id: int,
    ) -> int:
        """
        Delete all chunks of a document. Returns the deleted chunk count.
        """

        try:
            with self.session_factory() as db:
                with db.begin():
                    result = db.execute(
                        delete(DocumentChunk).where(
                            DocumentChunk.document_id == document_id
                        )
                    )
        except Exception as exc:
            raise RAGError(
                "knowledge base storage failed"
            ) from exc

        return result.rowcount
