from sqlalchemy import select
from sqlalchemy.orm import Session

from rag.models import DocumentChunk


class VectorRetriever:
    def __init__(self, db: Session):
        self.db = db

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: int | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        if not query_embedding:
            raise ValueError("query_embedding must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if score_threshold is not None and not 0 <= score_threshold <= 1:
            raise ValueError(
                "score_threshold must be between zero and one"
            )

        distance = DocumentChunk.embedding.cosine_distance(
            query_embedding
        )

        score = 1 - distance

        statement = select(
            DocumentChunk,
            distance.label("distance"),
        )

        if document_id is not None:
            statement = statement.where(
                DocumentChunk.document_id == document_id
            )

        if score_threshold is not None:
            statement = statement.where(
                score >= score_threshold
            )

        statement = (
            statement
            .order_by(distance)
            .limit(top_k)
        )

        rows = self.db.execute(statement).all()

        return [
            {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "score": 1 - float(distance_value),
            }
            for chunk, distance_value in rows
        ]