from pydantic import BaseModel, ConfigDict


class RAGSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: int | None
    chunk_index: int
    score: float


class RAGContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    sources: list[RAGSource]


class RAGContextBuilder:
    def build(
        self,
        results: list[dict],
    ) -> RAGContext:
        context_parts: list[str] = []
        sources: list[RAGSource] = []

        for index, result in enumerate(results, start=1):
            context_parts.append(
                f"[Source {index} | "
                f"Document {result['document_id']} | "
                f"Chunk {result['chunk_index']} | "
                f"Score {result['score']:.3f}]\n"
                f"{result['content']}"
            )

            sources.append(
                RAGSource(
                    document_id=result["document_id"],
                    chunk_index=result["chunk_index"],
                    score=result["score"],
                )
            )

        return RAGContext(
            text="\n\n".join(context_parts),
            sources=sources,
        )