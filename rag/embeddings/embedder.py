from typing import Protocol

from openai import OpenAI

from observability import record_embedding_usage, timed


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


class OpenAIEmbedder:
    def __init__(
        self,
        client: OpenAI,
        model: str = "text-embedding-3-small",
    ):
        self.client = client
        self.model = model

    def embed(self, text: str) -> list[float]:
        with timed("embedding_call"):
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )

        record_embedding_usage(
            getattr(response, "usage", None)
        )

        return response.data[0].embedding