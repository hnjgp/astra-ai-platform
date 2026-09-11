import pytest

from rag.chunking.chunker import DocumentChunker


def test_chunker_splits_text():

    chunker = DocumentChunker(
        chunk_size=10,
        chunk_overlap=0,
    )

    text = "ABCDEFGHIJ1234567890"

    result = chunker.split(text)

    assert result == [
        "ABCDEFGHIJ",
        "1234567890",
    ]


def test_chunker_creates_overlap():

    chunker = DocumentChunker(
        chunk_size=10,
        chunk_overlap=3,
    )

    text = "ABCDEFGHIJ1234567890"

    result = chunker.split(text)

    assert result == [
        "ABCDEFGHIJ",
        "HIJ1234567",
        "567890",
    ]


def test_chunker_returns_empty_list_for_empty_text():

    chunker = DocumentChunker()

    result = chunker.split("")

    assert result == []


def test_chunker_keeps_short_text_as_one_chunk():

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=10,
    )

    text = "Short document."

    result = chunker.split(text)

    assert result == [
        "Short document."
    ]


def test_chunker_rejects_invalid_chunk_size():

    with pytest.raises(
        ValueError,
        match="chunk_size must be greater than zero",
    ):
        DocumentChunker(
            chunk_size=0
        )


def test_chunker_rejects_negative_overlap():

    with pytest.raises(
        ValueError,
        match="chunk_overlap cannot be negative",
    ):
        DocumentChunker(
            chunk_size=100,
            chunk_overlap=-1,
        )


def test_chunker_rejects_overlap_equal_to_chunk_size():

    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        DocumentChunker(
            chunk_size=100,
            chunk_overlap=100,
        )