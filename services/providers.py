from functools import lru_cache

from openai import OpenAI

from config import OPENAI_API_KEY
from rag.database import RAGSessionLocal
from rag.embeddings.cache import CachedEmbedder
from rag.embeddings.embedder import Embedder, OpenAIEmbedder
from rag.indexer import DocumentIndexer


@lru_cache
def get_embedder() -> Embedder:
    return CachedEmbedder(
        OpenAIEmbedder(
            OpenAI(api_key=OPENAI_API_KEY)
        )
    )


@lru_cache
def get_document_indexer() -> DocumentIndexer:
    return DocumentIndexer(
        embedder=get_embedder(),
        session_factory=RAGSessionLocal,
    )
