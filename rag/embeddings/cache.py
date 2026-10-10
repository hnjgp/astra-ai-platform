import threading
from collections import OrderedDict

from observability import metrics
from rag.embeddings.embedder import Embedder


class CachedEmbedder:
    """
    LRU cache in front of an embedder, for query embeddings.

    Safe to cache: the same text with the same model always gives the
    same vector, and the vector holds no permission information.

    Deliberately not cached elsewhere:
    - model answers: depend on the user, permissions and conversation
    - retrieval results: change when documents are indexed or removed
    """

    def __init__(
        self,
        embedder: Embedder,
        max_size: int = 1_024,
    ):
        if max_size <= 0:
            raise ValueError(
                "max_size must be greater than zero"
            )

        self.embedder = embedder
        self.max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()

    def embed(self, text: str) -> list[float]:
        with self._lock:
            cached = self._cache.get(text)

            if cached is not None:
                self._cache.move_to_end(text)
                metrics.increment("embedding_cache", result="hit")
                return list(cached)

        metrics.increment("embedding_cache", result="miss")

        embedding = self.embedder.embed(text)

        with self._lock:
            self._cache[text] = list(embedding)
            self._cache.move_to_end(text)

            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

        return embedding

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        # Document chunks are rarely repeated; pass them through.
        return self.embedder.embed_many(texts)

    def __len__(self) -> int:
        return len(self._cache)
