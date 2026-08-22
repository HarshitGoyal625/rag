"""Typed intermediate representation (IR) of a parsed document.

Readers convert source files (PDF, DOCX, TXT, ...) into a `DocumentStructure`
instance. The chunker then walks this IR to produce chunks.

Design notes:
- Readers populate whatever structure they can detect; unknown/missing
  fields are left empty (e.g. a plain-txt file has no real headings).
- `to_dict()` / `from_dict()` are provided so the IR can be serialized to
  JSON for caching or debugging without giving up type safety in-process.
- `BlockType` is an enum so the chunker can switch on it cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Protocol
import hashlib


class BlockType(str, Enum):
    """The kind of content a `Block` represents.

    Inherits `str` so values serialize naturally to JSON
    (e.g. `"heading"` instead of `"BlockType.HEADING"`).
    """

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    PAGE_BREAK = "page_break"
    UNKNOWN = "unknown"


@dataclass
class Block:
    """A single structural unit of a document (heading, paragraph, table, ...).

    Attributes:
        type: The `BlockType` discriminator the chunker switches on.
        text: Flat text representation of the block (always populated).
        level: Heading level (1 = top-level). None for non-heading blocks.
        page: 1-indexed page number the block appears on. None if unknown.
        rows: For TABLE blocks: a 2D list of cell strings. None otherwise.
        extra: Free-form per-reader metadata (e.g. font size, style name).
    """

    type: BlockType
    text: str = ""
    level: Optional[int] = None
    page: Optional[int] = None
    rows: Optional[list[list[str]]] = None
    extra: dict = field(default_factory=dict)


@dataclass
class DocumentStructure:
    """Structured representation of a parsed document.

    A flat list of `Block`s is enough for most chunking strategies and
    keeps the IR versionable (just bump a `schema_version` if it grows).
    Blocks preserve source order; headings carry `level` so a chunker can
    rebuild a section tree or tuple-each-chunk with its ancestor headings.

    Attributes:
        source: Path/identifier the structure was parsed from.
        doc_type: Short tag for the source format ("pdf", "docx", ...).
        blocks: Ordered content blocks (headings, paragraphs, tables, ...).
        schema_version: IR version; bump when adding fields so old caches
            can be detected and rejected/regenerated.
    """

    source: str
    doc_type: str
    blocks: list[Block] = field(default_factory=list)
    schema_version: str = "1.0"
    metadata: dict = field(default_factory=dict)

    # ---- traversal helpers -------------------------------------------------

    def iter_headings(self) -> list[Block]:
        """Return only heading blocks (convenience for chunkers/debugging)."""
        return [b for b in self.blocks if b.type is BlockType.HEADING]

    def ancestors_for(self, block: Block) -> list[Block]:
        """walks from the beginning of the document to block, maintaining a stack of currently active headings, and returns the heading hierarchy that contains that block.
        """
        ancestors: list[Block] = []
        for b in self.blocks:
            if b is block:
                # Reached the target block itself; stop.
                break
            if b.type is not BlockType.HEADING:
                continue
            # Pop deeper/same-level headings before pushing a new one so the
            # list always represents the current open section hierarchy.
            while ancestors and b.level is not None and ancestors[-1].level >= b.level:
                ancestors.pop()
            ancestors.append(b)

        if block.type is BlockType.HEADING:
            # `block` is itself a heading; drop same-or-deeper headings so we
            # return its *parent* sections, not itself.
            while ancestors and ancestors[-1].level is not None and ancestors[-1].level >= (block.level or 0):
                ancestors.pop()
        return ancestors

    # ---- (de)serialization -------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "doc_type": self.doc_type,
            "metadata": self.metadata or {},
            "blocks": [_block_to_dict(b) for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentStructure":
        """Reconstruct a `DocumentStructure` from `to_dict()` output."""
        return cls(
            source=data["source"],
            doc_type=data["doc_type"],
            blocks=[_block_from_dict(b) for b in data.get("blocks", [])],
            schema_version=data.get("schema_version", "1.0"),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Serialize to a JSON string (use for caching/debugging only)."""
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ---- private block (de)serializers ------------------------------------------


def _block_to_dict(block: Block) -> dict:
    d = asdict(block)
    # `type` is a str Enum -> store its value ("heading", ...)
    d["type"] = block.type.value
    # Drop `None`/empty fields to keep payloads compact for caching.
    if block.rows is None:
        d.pop("rows")
    if not block.extra:
        d.pop("extra")
    if block.level is None:
        d.pop("level")
    if block.page is None:
        d.pop("page")
    if not block.text:
        d.pop("text")
    return d


def _block_from_dict(d: dict) -> Block:
    return Block(
        type=BlockType(d["type"]),
        text=d.get("text", ""),
        level=d.get("level"),
        page=d.get("page"),
        rows=d.get("rows"),
        extra=d.get("extra", {}),
    )


# =============================================================================
# Chunking Layer IR
# =============================================================================


class TokenCounter(Protocol):
    """Protocol for token counting implementations."""

    def count(self, text: str) -> int:
        ...


class TiktokenCounter:
    """Token counter using tiktoken (accurate, model-aware)."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "tiktoken is required for TiktokenCounter. "
                "Install with: pip install tiktoken"
            )
        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


class CharEstimator:
    """Fallback token estimator (~4 chars/token for English)."""

    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        return max(1, int(len(text) / self.chars_per_token))


@dataclass
class ChunkerConfig:
    """Configuration for the Block-Aware Recursive Chunker."""

    max_tokens: int = 512
    overlap_tokens: int = 50
    heading_context: bool = True
    table_row_limit: int = 50
    split_long_paragraphs: bool = True
    # min_tokens is a packing preference, not a hard filter
    min_tokens: int = 50


def _deterministic_chunk_id(
    source: str,
    block_indices: list[int],
    text: str,
) -> str:
    """Generate deterministic chunk ID from source + block range + content hash."""
    if not block_indices:
        block_range = "empty"
    elif len(block_indices) == 1:
        block_range = str(block_indices[0])
    else:
        block_range = f"{block_indices[0]}-{block_indices[-1]}"
    content_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    # Normalize source path for cross-platform consistency
    normalized_source = str(Path(source).as_posix())
    return f"{normalized_source}::{block_range}::{content_hash}"


@dataclass
class Chunk:
    """A retrieval unit produced by the chunker.

    Attributes:
        text: The chunk content (clean, without prepended heading context).
        chunk_id: Deterministic ID derived from source + block range + content.
        source: File path/identifier the chunk originated from.
        doc_type: Source format tag ("pdf", "docx", "txt", ...).
        page_range: (start_page, end_page) if pages known, else None.
        heading_path: Ancestor headings from root to this chunk's section.
        block_types: Block types contained in this chunk.
        block_indices: Indices into DocumentStructure.blocks that produced this chunk.
        token_count: Estimated token count of `text`.
        metadata: Extra info (is_continuation, table_rows, etc.).
    """

    text: str
    chunk_id: str
    source: str
    doc_type: str
    page_range: Optional[tuple[int, int]]
    heading_path: list[str]
    block_types: list[BlockType]
    block_indices: list[int]
    token_count: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "doc_type": self.doc_type,
            "page_range": self.page_range,
            "heading_path": self.heading_path,
            "block_types": [bt.value for bt in self.block_types],
            "block_indices": self.block_indices,
            "token_count": self.token_count,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Reconstruct from `to_dict()` output."""
        return cls(
            text=data["text"],
            chunk_id=data["chunk_id"],
            source=data["source"],
            doc_type=data["doc_type"],
            page_range=tuple(data["page_range"]) if data.get("page_range") else None,
            heading_path=data.get("heading_path", []),
            block_types=[BlockType(v) for v in data.get("block_types", [])],
            block_indices=data.get("block_indices", []),
            token_count=data.get("token_count", 0),
            metadata=data.get("metadata", {}),
        )
