from rag.context import RAGContextBuilder
from rag.embeddings.embedder import Embedder
from rag.prompt import RAGPromptBuilder
from rag.retrieval.service import RetrievalService


class RAGService:
    def __init__(
        self,
        embedder: Embedder,
        retrieval_service: RetrievalService,
        context_builder: RAGContextBuilder | None = None,
        prompt_builder: RAGPromptBuilder | None = None,
    ):
        self.embedder = embedder
        self.retrieval_service = retrieval_service
        self.context_builder = (
            context_builder or RAGContextBuilder()
        )
        self.prompt_builder = (
            prompt_builder or RAGPromptBuilder()
        )

    def build_prompt(
        self,
        question: str,
        top_k: int = 5,
        document_id: int | None = None,
        score_threshold: float | None = None,
    ) -> str:
        query_embedding = self.embedder.embed(question)

        retrieval_results = self.retrieval_service.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
            score_threshold=score_threshold,
        )

        context = self.context_builder.build(
            retrieval_results
        )

        return self.prompt_builder.build(context)