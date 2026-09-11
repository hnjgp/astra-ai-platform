from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from rag.pipeline import RAGPipeline


UPLOAD_DIRECTORY = Path("documents")

ALLOWED_EXTENSIONS = {
    ".txt",
    ".pdf",
}


def save_uploaded_document(
    upload_file: UploadFile,
) -> Path:
    """
    Save an uploaded document with a unique filename.
    """

    if not upload_file.filename:
        raise ValueError(
            "Uploaded file must have a filename"
        )

    original_path = Path(
        upload_file.filename
    )

    extension = (
        original_path.suffix.lower()
    )

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported document type"
        )

    UPLOAD_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_filename = (
        f"{uuid4().hex}{extension}"
    )

    destination = (
        UPLOAD_DIRECTORY
        / safe_filename
    )

    with destination.open(
        "wb"
    ) as file:

        while True:
            chunk = upload_file.file.read(
                1024 * 1024
            )

            if not chunk:
                break

            file.write(chunk)

    return destination


def process_uploaded_document(
    file_path: Path,
) -> tuple[str, list[str]]:
    """
    Process an uploaded document through the RAG pipeline.
    """

    pipeline = RAGPipeline()

    text = pipeline.loader.load(
        file_path
    )

    chunks = pipeline.chunker.split(
        text
    )

    return text, chunks