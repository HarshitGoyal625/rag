# RAG Architecture

A modular document processing pipeline: **Readers → IR → Chunkers → Embeddings**.

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐     ┌─────────────┐
│  Documents  │────▶│  DocumentReader  │────▶│  DocumentStructure │────▶│  Chunkers   │────▶ Chunks[]
│  (pdf,docx, │     │  Factory +       │     │  (Typed IR)        │     │  (Strategy) │
│   txt,...)  │     │  Concrete Readers│     │  Block[], metadata │     │             │
└─────────────┘     └──────────────────┘     └────────────────────┘     └─────────────┘
```

## Components

| Layer | File | Role |
|-------|------|------|
| **Entry** | `main.py` | Recursively reads dir, runs readers + chunkers, prints summaries |
| **IR** | `document_readers/structure.py` | Typed IR: `Block`, `BlockType`, `DocumentStructure`, `Chunk`, `ChunkerConfig`, token counters |
| **Readers** | `document_readers/base.py` | Abstract `DocumentReader` interface |
| | `document_readers/factory.py` | `DocumentReaderFactory` — routes by extension |
| | `document_readers/txt_reader.py` | `.txt`, `.md` → ATX headings + paragraphs |
| | `document_readers/pdf_reader.py` | `.pdf` via **PyMuPDF** — simple page-level text extraction (no complex heuristics) |
| | `document_readers/docx_reader.py` | `.docx` via `python-docx` — styles → headings, lists, tables |
| **Chunkers** | `chunkers/base.py` | Abstract `Chunker`, `ChunkerFactory` |
| | `chunkers/block_aware_recursive.py` | **Block-Aware Recursive Chunker** (default) |

---

## Document IR (`document_readers/structure.py`)

```python
# --- Blocks (reader output) ---
@dataclass
class Block:
    type: BlockType           # HEADING, PARAGRAPH, LIST_ITEM, TABLE, PAGE_BREAK, UNKNOWN
    text: str
    level: int | None         # heading level 1-6
    page: int | None          # 1-indexed
    rows: list[list[str]] | None  # TABLE only
    extra: dict               # reader-specific metadata

@dataclass
class DocumentStructure:
    source: str
    doc_type: str             # "pdf", "docx", "txt", ...
    blocks: list[Block]
    metadata: dict            # doc-level (title, author, dates)
    schema_version: str = "1.0"

    def iter_headings(self) -> list[Block]: ...
    def ancestors_for(self, block: Block) -> list[Block]: ...  # section stack
    def to_json(self) -> str: ...
    @classmethod
    def from_dict(cls, data: dict) -> "DocumentStructure": ...

# --- Chunking IR ---
@dataclass
class ChunkerConfig:
    max_tokens: int = 512
    overlap_tokens: int = 50
    heading_context: bool = True
    table_row_limit: int = 50
    split_long_paragraphs: bool = True
    min_tokens: int = 50      # packing preference, not hard filter

class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

@dataclass
class Chunk:
    text: str
    chunk_id: str             # deterministic: source::block_range::content_hash
    source: str
    doc_type: str
    page_range: tuple[int, int] | None
    heading_path: list[str]   # ancestor headings ["H1", "H2", ...]
    block_types: list[BlockType]
    block_indices: list[int]  # indices into DocumentStructure.blocks
    token_count: int
    metadata: dict            # is_continuation, chunk_index, table_rows, ...
```

---

## HLD: Chunking System

```
DocumentStructure (blocks[])
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│  Block-Aware Recursive Chunker                             │
├────────────────────────────────────────────────────────────┤
│  Phase 1: Boundary Determination (block index ranges)      │
│  • Walk blocks in order, maintain heading stack            │
│  • Greedy pack: accumulate blocks until max_tokens         │
│  • HEADING → update heading stack (prefer attach to next)  │
│  • Oversized block → split (table by rows, text by sents)  │
│  • Output: [(start_idx, end_idx), ...]                     │
├────────────────────────────────────────────────────────────┤
│  Phase 2: Build Chunks with Overlap                        │
│  • For each boundary: collect blocks → text                │
│  • heading_path = ancestors_for(first_block)               │
│  • page_range = min/max page from blocks                   │
│  • Apply token overlap from previous chunk                 │
│  • Deterministic chunk_id = hash(source + blocks + text)   │
│  • Emit Chunk[]                                            │
└────────────────────────────────────────────────────────────┘
        │
        ▼
Chunk[]  (clean text, rich metadata, no format awareness)
```

**Key Principles**
- **Format-independent**: Chunker never knows PDF vs DOCX vs TXT — only sees `BlockType`, `level`, `page`
- **Exploit structure when available**: Headings drive context; tables stay intact; lists pack together
- **Recursive split only when necessary**: Oversized block → sentences → words (fallback)
- **Token budget is the constraint**: `max_tokens`, `overlap_tokens` via pluggable `TokenCounter`
- **Deterministic IDs**: Re-ingestion produces same IDs → enables incremental indexing

---

## LLD: Block-Aware Recursive Chunker

### Class: `BlockAwareRecursiveChunker(Chunker)`

```python
def __init__(self, config: ChunkerConfig = None, token_counter: TokenCounter = None):
    # Uses TiktokenCounter (cl100k_base) if tiktoken available,
    # otherwise CharEstimator (~4 chars/token)

def chunk(self, doc: DocumentStructure) -> list[Chunk]:
    boundaries = self._determine_boundaries(doc)   # Phase 1
    return self._build_chunks(doc, boundaries)     # Phase 2
```

### Phase 1: `_determine_boundaries(doc) -> list[tuple[int, int]]`

```
heading_stack = []
current_start = 0, current_tokens = 0

for i, block in enumerate(doc.blocks):
    if block.type == HEADING:
        _update_heading_stack(heading_stack, block)  # pop deeper, push new

    block_tokens = token_counter.count(block.text)

    # Flush if adding block exceeds budget
    if current_tokens + block_tokens > max_tokens and current_tokens > 0:
        boundaries.append((current_start, i-1))
        current_start, current_tokens = i, 0

    # Oversized single block
    if block_tokens > max_tokens:
        if current_tokens > 0:
            boundaries.append((current_start, i-1))
        boundaries.extend(_split_oversized_block(doc, i, heading_stack))
        current_start, current_tokens = i+1, 0
        continue

    current_tokens += block_tokens

if current_start < len(doc.blocks):
    boundaries.append((current_start, len(doc.blocks)-1))
```

### `_split_oversized_block(doc, idx, heading_stack) -> list[tuple[int,int]]`

| BlockType | Strategy |
|-----------|----------|
| TABLE (has rows) | Split by data rows, repeat header row, respect token budget + `table_row_limit` |
| PARAGRAPH / UNKNOWN | Split by sentences (regex `(?<=[.!?])\s+`) |
| HEADING / LIST_ITEM / CODE | Emit as-is (don't split) |

### Phase 2: `_build_chunks(doc, boundaries) -> list[Chunk]`

```
for chunk_idx, (start, end) in enumerate(boundaries):
    blocks = doc.blocks[start:end+1]
    heading_path = [a.text for a in doc.ancestors_for(doc.blocks[start])] if heading_context else []
    page_range = (min(pages), max(pages)) from blocks with page != None
    text = "\n\n".join(block.text or table_to_pipes(block) for block in blocks)

    # Overlap from previous chunk
    if chunk_idx > 0 and overlap_tokens > 0:
        overlap = _get_overlap_text(prev_chunk.text)  # binary search by token count
        text = overlap + "\n\n" + text

    chunk_id = _deterministic_chunk_id(doc.source, list(range(start, end+1)), text)
    token_count = token_counter.count(text)

    Chunk(text, chunk_id, doc.source, doc.doc_type, page_range,
          heading_path, [b.type for b in blocks], list(range(start, end+1)),
          token_count, metadata)
```

### Overlap: `_get_overlap_text(prev_text) -> str`

Binary search for largest suffix ≤ `overlap_tokens` using the same `TokenCounter`.

---

## Changes Summary

| Area | Before | After |
|------|--------|-------|
| **PDF Reader** | Complex font heuristic, heading detection, table detection, running header detection | Simple page-level text extraction (1 paragraph block per page) |
| **Chunking** | Not implemented | **Block-Aware Recursive Chunker** with deterministic IDs, token overlap, heading context, table splitting |
| **Token Counting** | N/A | Pluggable `TokenCounter` protocol (tiktoken + fallback) |
| **IR** | `Block`, `DocumentStructure` | + `Chunk`, `ChunkerConfig`, `TokenCounter`, deterministic ID util |

---

## Installation

```bash
pip install -r requirements.txt
# Core: pymupdf, python-docx
# Optional (for accurate tokens): pip install tiktoken
```

## Usage

```bash
python main.py
# Reads ./documents, chunks with block_aware_recursive (max_tokens=512, overlap=50)
# Prints per-doc block summary + chunk summary
```

## Extending

**New Reader**: Subclass `DocumentReader` → implement `read()`, `supported_extensions()` → auto-registered via `__init__.py`.

**New Chunker**: Subclass `Chunker` → implement `chunk(doc) -> list[Chunk]` → decorate with `@ChunkerFactory.register("name")`.

---

## Next Steps

- Embedding layer: vectorize `Chunk.text` (optionally prepend `heading_path`)
- Indexing: store chunks with `chunk_id`, `source`, `page_range`, `metadata`
- Retrieval: hybrid search (dense + BM25) with citations from `page_range` + `heading_path`
- Evaluation: chunk quality metrics (size dist, heading coverage, table integrity)