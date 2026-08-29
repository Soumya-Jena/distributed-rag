from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
}


def load_text_file(path: Path) -> str:
    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def load_pdf(path: Path) -> str:
    reader = PdfReader(path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages)


def load_document(path: Path) -> str:

    extension = path.suffix.lower()

    if extension in {".txt", ".md"}:
        return load_text_file(path)

    if extension == ".pdf":
        return load_pdf(path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


def discover_documents(directory: Path):

    documents = []

    for path in directory.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            documents.append(path)

    return sorted(documents)