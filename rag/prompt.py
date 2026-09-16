from rag.context import RAGContext


class RAGPromptBuilder:
    def build(self, context: RAGContext) -> str:
        if not context.text or not context.sources:
            return "No relevant information was found in the knowledge base."

        source_blocks = []

        for index, source in enumerate(context.sources, start=1):
            source_blocks.append(
                f"[Source {index}]\n"
                f"Document ID: {source.document_id}\n"
                f"Chunk index: {source.chunk_index}\n"
            )

        sources_text = "\n".join(source_blocks)

        instructions = (
            "Use the provided knowledge base context to answer the "
            "user's question.\n"
            "If the answer is not found in the context, say so.\n"
            "Do not invent information.\n\n"
        )

        return (
            f"{instructions}"
            "Relevant information from the knowledge base:\n\n"
            f"{context.text}\n\n"
            f"{sources_text}"
        )