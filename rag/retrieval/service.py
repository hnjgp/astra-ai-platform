from sqlalchemy.orm import Session

from rag.retrieval.reranker import KeywordReranker
from rag.retrieval.retriever import VectorRetriever


class RetrievalService:
    def __init__(
        self,
        db: Session,
        retriever: VectorRetriever | None = None,
        reranker: KeywordReranker | None = None,
    ):
        self.retriever = (
            retriever
            or VectorRetriever(db)
        )
        self.reranker = reranker

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: int | None = None,
        score_threshold: float | None = None,
        query: str | None = None,
    ) -> list[dict]:

        results = self.retriever.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
            score_threshold=score_threshold,
        )

        if self.reranker is None:
            return results

        if query is None:
            raise ValueError(
                "query is required when reranker is enabled"
            )

        return self.reranker.rerank(
            query=query,
            results=results,
        )