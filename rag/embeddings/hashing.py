import hashlib
import math
import re


class HashingEmbedder:
    """
    Deterministic bag-of-words embedder (feature hashing), 1536 dims.

    No API calls: used for offline evaluation and tests. It only
    captures shared words, not meaning, so it is a baseline, not a
    replacement for a real embedding model.
    """

    dimensions = 1536

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions

        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))

        if norm == 0:
            vector[0] = 1.0
            return vector

        return [value / norm for value in vector]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
