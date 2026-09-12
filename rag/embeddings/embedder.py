from typing import Protocol

from openai import OpenAI


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
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )

        return response.data[0].embedding