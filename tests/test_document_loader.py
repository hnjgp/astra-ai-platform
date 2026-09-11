from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from rag.ingestion.loader import DocumentLoader


def test_loader_reads_text_file(tmp_path: Path):

    file_path = tmp_path / "document.txt"

    file_path.write_text(
        "FastAPI is a Python web framework.",
        encoding="utf-8",
    )

    loader = DocumentLoader()

    result = loader.load(file_path)

    assert result == (
        "FastAPI is a Python web framework."
    )


def test_loader_extracts_text_from_pdf(
    tmp_path: Path,
):

    file_path = tmp_path / "document.pdf"

    pdf = canvas.Canvas(
        str(file_path)
    )

    pdf.drawString(
        100,
        750,
        "Astra RAG document.",
    )

    pdf.save()

    loader = DocumentLoader()

    result = loader.load(file_path)

    assert "Astra RAG document." in result


def test_loader_rejects_unsupported_file_type(
    tmp_path: Path,
):

    file_path = tmp_path / "document.docx"

    file_path.write_text(
        "unsupported",
        encoding="utf-8",
    )

    loader = DocumentLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported document type",
    ):
        loader.load(file_path)


def test_loader_raises_for_missing_file(
    tmp_path: Path,
):

    file_path = tmp_path / "missing.txt"

    loader = DocumentLoader()

    with pytest.raises(
        FileNotFoundError
    ):
        loader.load(file_path)


def test_loader_raises_for_directory(
    tmp_path: Path,
):

    directory = tmp_path / "documents"
    directory.mkdir()

    loader = DocumentLoader()

    with pytest.raises(
        ValueError,
        match="Path is not a file",
    ):
        loader.load(directory)


def test_loader_supports_uppercase_extension(
    tmp_path: Path,
):

    file_path = tmp_path / "document.TXT"

    file_path.write_text(
        "Astra RAG document.",
        encoding="utf-8",
    )

    loader = DocumentLoader()

    result = loader.load(file_path)

    assert result == "Astra RAG document."