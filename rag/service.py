from collections.abc import Collection

from rag.context import RAGContext, RAGContextBuilder
from rag.embeddings.embedder import Embedder
from rag.exceptions import RAGError
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

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        document_id: int | None = None,
        score_threshold: float | None = None,
        document_ids: Collection[int] | None = None,
    ) -> RAGContext:

        if not question.strip():
            raise ValueError(
                "question must not be empty"
            )

        access_filter = (
            {}
            if document_ids is None
            else {"document_ids": document_ids}
        )

        try:
            query_embedding = self.embedder.embed(
                question
            )

            retrieval_results = (
                self.retrieval_service.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    document_id=document_id,
                    score_threshold=score_threshold,
                    query=question,
                    **access_filter,
                )
            )

            return self.context_builder.build(
                retrieval_results
            )

        except ValueError:
            raise

        except RAGError:
            raise

        except Exception as exc:
            raise RAGError(
                "RAG processing failed"
            ) from exc

    def build_prompt(
        self,
        question: str,
        top_k: int = 5,
        document_id: int | None = None,
        score_threshold: float | None = None,
        document_ids: Collection[int] | None = None,
    ) -> str:

        context = self.retrieve(
            question=question,
            top_k=top_k,
            document_id=document_id,
            score_threshold=score_threshold,
            document_ids=document_ids,
        )

        try:
            return self.prompt_builder.build(
                context
            )

        except ValueError:
            raise

        except Exception as exc:
            raise RAGError(
                "RAG processing failed"
            ) from exc
