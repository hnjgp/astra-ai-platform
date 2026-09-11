from pathlib import Path


class DocumentLoader:
    """
    Load supported documents and return their text content.
    """

    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".pdf",
    }

    def load(self, file_path: str | Path) -> str:
        """
        Load a document and return its text content.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported document type: {extension}"
            )

        if extension == ".txt":
            return self._load_text(path)

        if extension == ".pdf":
            return self._load_pdf(path)

        raise ValueError(
            f"Unsupported document type: {extension}"
        )

    def _load_text(self, path: Path) -> str:
        """
        Load a plain text document.
        """

        return path.read_text(
            encoding="utf-8"
        )

    def _load_pdf(self, path: Path) -> str:
        """
        Extract text from all pages of a PDF document.
        """

        from pypdf import PdfReader

        reader = PdfReader(path)

        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n\n".join(pages)