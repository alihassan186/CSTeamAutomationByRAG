from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {path.name}")


def iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def load_documents(raw_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in iter_files(raw_dir):
        loaded = load_file(path)
        for doc in loaded:
            doc.metadata = dict(doc.metadata or {})
            doc.metadata.setdefault("source", str(path))
        docs.extend(loaded)
    return docs
