from rag.retrieval.reranker import KeywordReranker


def test_keyword_reranker_moves_more_relevant_result_to_top():
    results = [
        {
            "document_id": 1,
            "chunk_index": 0,
            "content": (
                "Python is a programming language."
            ),
            "score": 0.95,
        },
        {
            "document_id": 1,
            "chunk_index": 1,
            "content": (
                "FastAPI is a Python web framework "
                "for building APIs."
            ),
            "score": 0.90,
        },
    ]

    reranker = KeywordReranker()

    reranked = reranker.rerank(
        query=(
            "Which Python web framework "
            "should I use for an API?"
        ),
        results=results,
    )

    assert reranked[0]["chunk_index"] == 1
    assert reranked[1]["chunk_index"] == 0


def test_keyword_reranker_preserves_result_data():
    result = {
        "document_id": 1,
        "chunk_index": 3,
        "content": "FastAPI is a Python web framework.",
        "score": 0.88,
    }

    reranker = KeywordReranker()

    reranked = reranker.rerank(
        query="Python web framework",
        results=[result],
    )

    assert reranked == [result]


def test_keyword_reranker_returns_empty_results():
    reranker = KeywordReranker()

    reranked = reranker.rerank(
        query="Python framework",
        results=[],
    )

    assert reranked == []


def test_keyword_reranker_rejects_empty_query():
    reranker = KeywordReranker()

    try:
        reranker.rerank(
            query="",
            results=[],
        )
    except ValueError as exc:
        assert str(exc) == "query must not be empty"
    else:
        raise AssertionError(
            "ValueError was not raised"
        )