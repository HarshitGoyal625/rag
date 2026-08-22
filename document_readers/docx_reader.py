"""DOCX file reader implementation.

Reads a Word .docx into a structured `DocumentStructure` IR. Because DOCX
paragraphs carry an explicit style (`paragraph.style.name`), heading levels
are detected reliably when the author used Word's built-in heading styles.
Paragraphs without a recognised heading style become `PARAGRAPH` blocks;
tables become `TABLE` blocks. DOCX has no page concept, so `page` stays
None throughout.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from .base import DocumentReader
from .structure import Block, BlockType, DocumentStructure

if TYPE_CHECKING:
    from docx.text.paragraph import Paragraph
    from docx.table import Table
    from docx.document import Document as DocumentType

# "Heading 1" -> 1, "Heading 2" -> 2, ... Also tolerate localized/renamed
# styles by checking the style name starts with "Heading" + a number.
_HEADING_RE = re.compile(r"^Heading\s+(\d+)$", re.IGNORECASE)


def iter_block_items(parent: "DocumentType"):
    """Iterate over body-level Paragraph and Table objects in document order.

    Unlike ``doc.paragraphs`` and ``doc.tables`` which expose each type
    separately and lose interleaving order, this walks the XML body children
    directly and wraps each element into the appropriate python-docx object.
    """
    from docx.text.paragraph import Paragraph
    from docx.table import Table

    parent_element = parent.element.body
    for child in parent_element.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("}tbl"):
            yield Table(child, parent)


class DocxReader(DocumentReader):
    """Reader for Microsoft Word DOCX files using python-docx."""

    def read(self, file_path: Path) -> DocumentStructure:
        """Read a DOCX file and return it as a structured IR."""
        from docx import Document

        doc = Document(file_path)

        blocks: list[Block] = []

        from docx.text.paragraph import Paragraph
        from docx.table import Table

        for item in iter_block_items(doc):
            if isinstance(item, Paragraph):
                block = self._block_from_paragraph(item)
                if block is not None:
                    blocks.append(block)
            elif isinstance(item, Table):
                blocks.append(self._block_from_table(item))

        return DocumentStructure(
            source=str(file_path),
            doc_type="docx",
            blocks=blocks,
        )

    def _block_from_paragraph(self, paragraph) -> Block | None:
        """Convert a python-docx paragraph into a `Block`, or skip it.

        Returns None for empty paragraphs (skip). Heading styles are mapped
        to `HEADING` blocks with a numeric `level`; everything else becomes
        a `PARAGRAPH` block.
        """
        text = paragraph.text.strip()
        if not text:
            return None

        style_name = (paragraph.style.name or "") if paragraph.style else ""
        heading_match = _HEADING_RE.match(style_name)

        if heading_match:
            return Block(
                type=BlockType.HEADING,
                text=text,
                level=int(heading_match.group(1)),
                extra={"style": style_name},
            )

        # List styles typically start with "List" (e.g. "List Bullet").
        if style_name.lower().startswith("list"):
            return Block(
                type=BlockType.LIST_ITEM,
                text=text,
                extra={"style": style_name},
            )

        return Block(
            type=BlockType.PARAGRAPH,
            text=text,
            extra={"style": style_name} if style_name else {},
        )

    def _block_from_table(self, table) -> Block:
        """Convert a python-docx table into a `TABLE` block."""
        rows: list[list[str]] = []
        for row in table.rows:
            rows.append(
                [cell.text.strip() for cell in row.cells]
            )
        return Block(
            type=BlockType.TABLE,
            text="\n".join(" | ".join(r) for r in rows),
            rows=rows,
        )

    def supported_extensions(self) -> list[str]:
        """Return supported DOCX file extensions.

        Note: .doc (legacy Word) is NOT supported by python-docx. Use
        antiword, textract, or convert to .docx first.
        """
        return [".docx"]