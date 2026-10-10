import pytest

from observability import metrics
from rag.embeddings.cache import CachedEmbedder
from tests.fakes import FakeEmbedder


@pytest.fixture(autouse=True)
def clean_metrics():
    metrics.reset()
    yield
    metrics.reset()


def test_repeated_query_uses_cache():
    inner = FakeEmbedder()
    embedder = CachedEmbedder(inner)

    first = embedder.embed("what is pgvector")
    second = embedder.embed("what is pgvector")

    assert first == second
    assert inner.calls == 1
    assert metrics.counter("embedding_cache", result="hit") == 1
    assert metrics.counter("embedding_cache", result="miss") == 1


def test_different_queries_are_embedded_separately():
    inner = FakeEmbedder()
    embedder = CachedEmbedder(inner)

    embedder.embed("one")
    embedder.embed("two")

    assert inner.calls == 2


def test_least_recently_used_entry_is_evicted():
    inner = FakeEmbedder()
    embedder = CachedEmbedder(inner, max_size=2)

    embedder.embed("a")
    embedder.embed("b")
    embedder.embed("a")  # "a" is now most recently used
    embedder.embed("c")  # evicts "b"

    assert len(embedder) == 2

    embedder.embed("a")
    assert inner.calls == 3

    embedder.embed("b")
    assert inner.calls == 4


def test_cached_vector_cannot_be_changed_by_caller():
    embedder = CachedEmbedder(FakeEmbedder())

    vector = embedder.embed("text")
    vector[0] = 999.0

    assert embedder.embed("text")[0] != 999.0


def test_batch_embedding_is_not_cached():
    inner = FakeEmbedder()
    embedder = CachedEmbedder(inner)

    embedder.embed_many(["a", "b"])
    embedder.embed_many(["a", "b"])

    assert inner.batch_calls == 2
    assert len(embedder) == 0


def test_invalid_size_is_rejected():
    with pytest.raises(ValueError):
        CachedEmbedder(FakeEmbedder(), max_size=0)
