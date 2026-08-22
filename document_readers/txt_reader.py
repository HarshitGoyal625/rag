"""Text file reader implementation.

Reads plain-text and markdown files into a structured `DocumentStructure` IR.
For .md/.markdown we parse ATX-style headings (`#`, `##`, ...). For plain
text (.txt, .csv, .rst, .xml, .html, .htm, .json) we simply emit the whole
content as a single `PARAGRAPH` block — there is no reliable structure to
recover, and emitting paragraphs naively adds little chunking value.
"""
from __future__ import annotations

import re
from pathlib import Path

from .base import DocumentReader
from .structure import Block, BlockType, DocumentStructure

# ATX markdown headings: 1-6 leading '#' followed by space/text.
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# Files that benefit from markdown heading parsing.
_MARKDOWN_EXT = {".md", ".markdown"}


class TxtReader(DocumentReader):
    """Reader for plain text and markdown files."""

    def read(self, file_path: Path) -> DocumentStructure:
        """Read a text/markdown file and return it as a structured IR."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 for files with different encodings
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()

        ext = file_path.suffix.lower()
        blocks = (
            self._parse_markdown(content)
            if ext in _MARKDOWN_EXT
            else self._parse_plain(content)
        )

        return DocumentStructure(
            source=str(file_path),
            doc_type=ext.lstrip(".") or "txt",
            blocks=blocks,
        )

    def _parse_markdown(self, content: str) -> list[Block]:
        """Emit HEADING / PARAGRAPH blocks from markdown text.

        Headings (`#`..`######`) become `HEADING` blocks with the matching
        level; non-heading lines are grouped into a single paragraph until
        a heading or blank line breaks them.
        """
        blocks: list[Block] = []
        para_lines: list[str] = []

        def flush_paragraph() -> None:
            if para_lines:
                text = " ".join(para_lines).strip()
                if text:
                    blocks.append(Block(type=BlockType.PARAGRAPH, text=text))
                para_lines.clear()

        for line in content.splitlines():
            match = _MARKDOWN_HEADING_RE.match(line)
            if match:
                flush_paragraph()
                blocks.append(
                    Block(
                        type=BlockType.HEADING,
                        text=match.group(2).strip(),
                        level=len(match.group(1)),
                    )
                )
                continue
            if not line.strip():
                flush_paragraph()
                continue
            para_lines.append(line.strip())

        flush_paragraph()
        return blocks

    def _parse_plain(self, content: str) -> list[Block]:
        """Emit a single PARAGRAPH block for unstructured text."""
        text = content.strip()
        if not text:
            return []
        return [Block(type=BlockType.PARAGRAPH, text=text)]

    def supported_extensions(self) -> list[str]:
        """Return supported text file extensions."""
        return [".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".xml", ".html", ".htm"]
