from sqlalchemy.orm import Session

from rag.retrieval.retriever import VectorRetriever


class RetrievalService:
    def __init__(
        self,
        db: Session,
        retriever: VectorRetriever | None = None,
    ):
        self.retriever = retriever or VectorRetriever(db)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: int | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        return self.retriever.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
            score_threshold=score_threshold,
        )