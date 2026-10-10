from typing import Protocol

from openai import OpenAI

from observability import record_embedding_usage, timed


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbedder:
    # One request per batch instead of one per chunk; the API accepts
    # up to 2048 inputs per request.
    batch_size = 100

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

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]

            with timed("embedding_call"):
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )

            record_embedding_usage(
                getattr(response, "usage", None)
            )

            # The API returns one item per input, tagged with its index.
            ordered = sorted(
                response.data,
                key=lambda item: item.index,
            )

            embeddings.extend(
                item.embedding for item in ordered
            )

        return embeddings
