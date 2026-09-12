from dataclasses import dataclass


@dataclass
class EmbeddedChunk:
    text: str
    embedding: list[float]