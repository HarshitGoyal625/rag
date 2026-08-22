"""Block-Aware Recursive Chunker implementation.

Strategy:
1. Walk blocks in order, maintaining heading context stack
2. Greedily pack blocks into chunks up to max_tokens
3. For oversized blocks, recursively split them (sentences -> words)
4. After boundary determination, apply token-based overlap
5. Emit Chunk objects with rich metadata
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Union

from document_readers.structure import (
    Block,
    BlockType,
    Chunk,
    ChunkerConfig,
    DocumentStructure,
    TokenCounter,
    TiktokenCounter,
    CharEstimator,
    _deterministic_chunk_id,
)


@dataclass
class ChunkBoundary:
    """Represents a chunk boundary - either a block range or a split text segment."""
    start_idx: int
    end_idx: int
    # For split oversized blocks: the actual text segment (None = use full blocks)
    split_text: Optional[str] = None
    # For split tables: the rows included (None = use full block)
    table_rows: Optional[list[list[str]]] = None

    def is_split(self) -> bool:
        return self.split_text is not None or self.table_rows is not None


class BlockAwareRecursiveChunker:
    """Chunk a DocumentStructure by packing blocks, splitting only when necessary."""

    def __init__(
        self,
        config: Optional[ChunkerConfig] = None,
        token_counter: Optional[TokenCounter] = None,
    ):
        self.config = config or ChunkerConfig()
        # Initialize token counter (try tiktoken, fall back to char estimator)
        if token_counter is not None:
            self._token_counter = token_counter
        else:
            try:
                self._token_counter = TiktokenCounter()
            except ValueError:
                self._token_counter = CharEstimator()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def chunk(self, doc: DocumentStructure) -> list[Chunk]:
        """Split document into chunks."""
        if not doc.blocks:
            return []

        # Phase 1: Determine chunk boundaries
        boundaries = self._determine_boundaries(doc)

        # Phase 2: Build chunks with overlap
        chunks = self._build_chunks(doc, boundaries)

        return chunks

    # -----------------------------------------------------------------------
    # Phase 1: Boundary Determination
    # -----------------------------------------------------------------------

    def _determine_boundaries(self, doc: DocumentStructure) -> list[ChunkBoundary]:
        """Determine chunk boundaries as ChunkBoundary objects."""
        boundaries: list[ChunkBoundary] = []
        current_start = 0
        current_tokens = 0
        heading_stack: list[Block] = []

        for i, block in enumerate(doc.blocks):
            block_tokens = self._token_counter.count(block.text)

            # Handle HEADING: flush chunk if same or higher level than current section
            if block.type is BlockType.HEADING:
                heading_level = block.level or 1
                current_section_level = self._get_current_section_level(heading_stack)
                
                if current_section_level is not None and heading_level <= current_section_level:
                    # Same or higher level heading -> new section, flush current chunk
                    if current_start < i:
                        boundaries.append(ChunkBoundary(start_idx=current_start, end_idx=i - 1))
                        current_start = i
                        current_tokens = 0
                
                # Update heading stack
                self._update_heading_stack(heading_stack, block)

            # Check if adding this block would exceed budget
            if current_tokens + block_tokens > self.config.max_tokens and current_tokens > 0:
                # Flush current chunk
                boundaries.append(ChunkBoundary(start_idx=current_start, end_idx=i - 1))
                current_start = i
                current_tokens = 0

            # Handle oversized single block
            if block_tokens > self.config.max_tokens:
                if current_tokens > 0:
                    # Flush what we have first
                    boundaries.append(ChunkBoundary(start_idx=current_start, end_idx=i - 1))
                    current_start = i
                    current_tokens = 0

                # Split oversized block into sub-chunks
                sub_boundaries = self._split_oversized_block(doc, i, heading_stack)
                boundaries.extend(sub_boundaries)
                # Next chunk starts after this block
                current_start = i + 1
                current_tokens = 0
                continue

            # Normal packing
            current_tokens += block_tokens

        # Don't forget the last chunk
        if current_start < len(doc.blocks):
            boundaries.append(ChunkBoundary(start_idx=current_start, end_idx=len(doc.blocks) - 1))

        return boundaries

    def _get_current_section_level(self, heading_stack: list[Block]) -> Optional[int]:
        """Get the level of the current active section (deepest heading in stack)."""
        if not heading_stack:
            return None
        # The current section is the deepest heading in the stack
        return heading_stack[-1].level or 1

    def _update_heading_stack(self, stack: list[Block], heading: Block) -> None:
        """Maintain stack of active headings (ancestors)."""
        level = heading.level or 1
        while stack and (stack[-1].level or 0) >= level:
            stack.pop()
        stack.append(heading)

    def _split_oversized_block(
        self,
        doc: DocumentStructure,
        block_idx: int,
        heading_stack: list[Block],
    ) -> list[ChunkBoundary]:
        """Split a single oversized block into multiple chunk boundaries with text segments."""
        block = doc.blocks[block_idx]
        text = block.text

        # Determine splitter based on block type
        if block.type is BlockType.TABLE and block.rows:
            return self._split_table(block, block_idx)
        elif block.type is BlockType.PARAGRAPH or block.type is BlockType.UNKNOWN:
            return self._split_text_block(text, block_idx)
        else:
            # HEADING, LIST_ITEM, CODE - don't split, emit as-is
            return [ChunkBoundary(start_idx=block_idx, end_idx=block_idx)]

    def _split_table(
        self,
        block: Block,
        block_idx: int,
    ) -> list[ChunkBoundary]:
        """Split a table by rows, repeating header row."""
        if not block.rows:
            return [ChunkBoundary(start_idx=block_idx, end_idx=block_idx)]

        # Assume first row is header
        header = block.rows[0] if block.rows else []
        data_rows = block.rows[1:] if len(block.rows) > 1 else []

        if not data_rows:
            return [ChunkBoundary(start_idx=block_idx, end_idx=block_idx)]

        sub_chunks: list[ChunkBoundary] = []
        current_rows = [header]
        current_tokens = self._token_counter.count(" | ".join(header))

        for row in data_rows:
            row_text = " | ".join(row)
            row_tokens = self._token_counter.count(row_text)

            if current_tokens + row_tokens > self.config.max_tokens and len(current_rows) > 1:
                # Emit current sub-chunk with its rows
                sub_chunks.append(ChunkBoundary(
                    start_idx=block_idx,
                    end_idx=block_idx,
                    table_rows=list(current_rows)
                ))
                current_rows = [header]
                current_tokens = self._token_counter.count(" | ".join(header))

            current_rows.append(row)
            current_tokens += row_tokens

            # Safety: hard row limit
            if len(current_rows) >= self.config.table_row_limit:
                sub_chunks.append(ChunkBoundary(
                    start_idx=block_idx,
                    end_idx=block_idx,
                    table_rows=list(current_rows)
                ))
                current_rows = [header]
                current_tokens = self._token_counter.count(" | ".join(header))

        if len(current_rows) > 1:
            sub_chunks.append(ChunkBoundary(
                start_idx=block_idx,
                end_idx=block_idx,
                table_rows=list(current_rows)
            ))

        return sub_chunks if sub_chunks else [ChunkBoundary(start_idx=block_idx, end_idx=block_idx)]

    def _split_text_block(self, text: str, block_idx: int) -> list[ChunkBoundary]:
        """Split a long text block by sentences, then words if needed."""
        # Split by sentences (simple regex)
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [ChunkBoundary(start_idx=block_idx, end_idx=block_idx)]

        sub_chunks: list[ChunkBoundary] = []
        current_sentences = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._token_counter.count(sent)

            if current_tokens + sent_tokens > self.config.max_tokens and current_sentences:
                # Emit current sub-chunk with its text segment
                sub_chunks.append(ChunkBoundary(
                    start_idx=block_idx,
                    end_idx=block_idx,
                    split_text=" ".join(current_sentences)
                ))
                current_sentences = []
                current_tokens = 0

            current_sentences.append(sent)
            current_tokens += sent_tokens

        if current_sentences:
            sub_chunks.append(ChunkBoundary(
                start_idx=block_idx,
                end_idx=block_idx,
                split_text=" ".join(current_sentences)
            ))

        return sub_chunks if sub_chunks else [ChunkBoundary(start_idx=block_idx, end_idx=block_idx)]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences (simple but effective)."""
        # Split on sentence boundaries, keeping the delimiter
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p]

    # -----------------------------------------------------------------------
    # Phase 2: Build Chunks with Overlap
    # -----------------------------------------------------------------------

    def _build_chunks(
        self,
        doc: DocumentStructure,
        boundaries: list[ChunkBoundary],
    ) -> list[Chunk]:
        """Construct Chunk objects from boundaries with token overlap."""
        chunks: list[Chunk] = []

        for chunk_idx, boundary in enumerate(boundaries):
            # Get heading context for this chunk (use first block)
            heading_path = self._get_heading_path(doc, boundary.start_idx)

            # Build text content
            if boundary.is_split():
                if boundary.split_text is not None:
                    chunk_text = boundary.split_text
                elif boundary.table_rows is not None:
                    chunk_text = "\n".join(" | ".join(row) for row in boundary.table_rows)
                else:
                    chunk_text = ""
            else:
                # Normal block range
                block_indices = list(range(boundary.start_idx, boundary.end_idx + 1))
                blocks = [doc.blocks[i] for i in block_indices]
                chunk_text = self._blocks_to_text(blocks)

            # Apply overlap from previous chunk (if not first)
            if chunk_idx > 0 and self.config.overlap_tokens > 0:
                prev_chunk_text = chunks[-1].text
                overlap_text = self._get_overlap_text(prev_chunk_text)
                if overlap_text:
                    chunk_text = overlap_text + "\n\n" + chunk_text

            # Determine block indices for this chunk
            if boundary.is_split():
                block_indices = [boundary.start_idx]
            else:
                block_indices = list(range(boundary.start_idx, boundary.end_idx + 1))

            # Collect blocks for page range and types
            blocks = [doc.blocks[i] for i in block_indices]

            # Compute page range
            page_range = self._compute_page_range(blocks)

            # Collect block types
            block_types = [b.type for b in blocks]

            # Token count
            token_count = self._token_counter.count(chunk_text)

            # Deterministic ID
            chunk_id = _deterministic_chunk_id(
                doc.source,
                block_indices,
                chunk_text,
            )

            # Metadata
            metadata = {
                "is_continuation": chunk_idx > 0,
                "chunk_index": chunk_idx,
                "source": doc.source,
                "file_type": doc.doc_type,
            }
            # Add table metadata if applicable
            if boundary.table_rows is not None:
                metadata["table_rows"] = len(boundary.table_rows)
                metadata["table_split"] = True
            elif boundary.split_text is not None:
                metadata["text_split"] = True

            chunk = Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source=doc.source,
                doc_type=doc.doc_type,
                page_range=page_range,
                heading_path=heading_path,
                block_types=block_types,
                block_indices=block_indices,
                token_count=token_count,
                metadata=metadata,
            )
            chunks.append(chunk)

        return chunks

    def _get_heading_path(self, doc: DocumentStructure, block_idx: int) -> list[str]:
        """Get ancestor heading path for a block index."""
        if not self.config.heading_context:
            return []

        block = doc.blocks[block_idx]
        ancestors = doc.ancestors_for(block)
        return [a.text for a in ancestors]

    def _compute_page_range(
        self,
        blocks: list[Block],
    ) -> Optional[tuple[int, int]]:
        """Compute (min_page, max_page) from blocks, ignoring None."""
        pages = [b.page for b in blocks if b.page is not None]
        if not pages:
            return None
        return (min(pages), max(pages))

    def _blocks_to_text(self, blocks: list[Block]) -> str:
        """Convert blocks to a single text string."""
        parts = []
        for block in blocks:
            if block.type is BlockType.TABLE and block.rows:
                # Render table as pipe-separated rows
                table_text = "\n".join(" | ".join(row) for row in block.rows)
                parts.append(table_text)
            else:
                parts.append(block.text)
        return "\n\n".join(parts)

    def _get_overlap_text(self, prev_text: str) -> str:
        """Extract last N tokens from previous chunk for overlap."""
        if not prev_text:
            return ""

        # Estimate how many characters correspond to overlap_tokens
        # Using the same counter for consistency
        target_tokens = self.config.overlap_tokens

        # Binary search for character position
        left, right = 0, len(prev_text)
        best = 0

        while left <= right:
            mid = (left + right) // 2
            candidate = prev_text[mid:]
            tokens = self._token_counter.count(candidate)
            if tokens <= target_tokens:
                best = mid
                right = mid - 1
            else:
                left = mid + 1

        overlap = prev_text[best:]
        return overlap if overlap else ""