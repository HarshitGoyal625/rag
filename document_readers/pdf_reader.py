"""PDF file reader implementation (PyMuPDF / pymupdf).

Simple, reliable text extraction with page boundaries preserved.
No complex structural analysis - just plain text per page.
"""
from __future__ import annotations

from pathlib import Path

from .base import DocumentReader
from .structure import Block, BlockType, DocumentStructure


class PdfReader(DocumentReader):
    """Reader for PDF files using PyMuPDF (pymupdf), simple text extraction."""

    def read(self, file_path: Path) -> DocumentStructure:
        """Read a PDF and return it as a structured IR with page-level blocks."""
        import pymupdf

        with pymupdf.open(file_path) as doc:
            blocks = self._extract_blocks(doc)
            metadata = self._extract_metadata(doc)

        return DocumentStructure(
            source=str(file_path),
            doc_type="pdf",
            blocks=blocks,
            metadata=metadata,
        )

    @staticmethod
    def _extract_metadata(doc) -> dict:
        """Pull non-empty `doc.metadata` entries (title, author, dates, ...)."""
        out: dict = {}
        md = doc.metadata or {}
        for k, v in md.items():
            if v not in (None, "", []):
                out[k] = v
        return out

    def _extract_blocks(self, doc) -> list[Block]:
        """Extract text blocks, one per page."""
        blocks: list[Block] = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            text = text.strip()
            if text:
                blocks.append(
                    Block(
                        type=BlockType.PARAGRAPH,
                        text=text,
                        page=page_num,
                        extra={"page": page_num},
                    )
                )

        return blocks

    def supported_extensions(self) -> list[str]:
        """Return supported PDF file extensions."""
        return [".pdf"]