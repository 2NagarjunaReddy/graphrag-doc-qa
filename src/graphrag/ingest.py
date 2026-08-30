from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents(docs_dir: str | Path) -> dict[str, str]:
    """Read every supported file in docs_dir into {doc_id: full_text}."""
    docs_dir = Path(docs_dir)
    documents: dict[str, str] = {}
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = _read_pdf_file(path) if path.suffix.lower() == ".pdf" else _read_text_file(path)
        if text.strip():
            documents[path.stem] = text
    return documents


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> list[str]:
    """Split text into overlapping word-count windows, breaking on sentence ends where possible."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks


def build_chunks(documents: dict[str, str], chunk_size: int = 220, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_id, text in documents.items():
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(Chunk(doc_id=doc_id, chunk_id=f"{doc_id}::{i}", text=piece))
    return chunks
