"""Test script to inspect chunking quality for DOCX and TXT files."""
from pathlib import Path
from document_readers import DocumentReaderFactory
from chunkers import BlockAwareRecursiveChunker
from document_readers.structure import ChunkerConfig


def test_file(file_path: Path, max_tokens: int = 200, overlap_tokens: int = 30):
    print(f"\n{'='*70}")
    print(f"TEST: {file_path.name}")
    print(f"{'='*70}")

    doc = DocumentReaderFactory.read_document(file_path)
    print(f"\nDocument: {doc.source}")
    print(f"Type: {doc.doc_type}")
    print(f"Blocks: {len(doc.blocks)}")

    # Show all blocks
    print(f"\n--- Blocks ---")
    for i, b in enumerate(doc.blocks):
        lvl = f" L{b.level}" if b.level else ""
        pg = f" p{b.page}" if b.page else ""
        print(f"  [{i}] {b.type.value}{lvl}{pg}: {b.text[:100]}...")

    # Chunk it
    config = ChunkerConfig(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    chunker = BlockAwareRecursiveChunker(config=config)
    chunks = chunker.chunk(doc)

    print(f"\n--- Chunks ({len(chunks)}) ---")
    for c in chunks:
        pg = f" p{c.page_range[0]}-{c.page_range[1]}" if c.page_range else ""
        hp = f"\n    heading_path: {c.heading_path}" if c.heading_path else ""
        print(f"\n  Chunk {c.metadata.get('chunk_index', '?')}: {c.token_count} tok{pg}{hp}")
        print(f"    blocks: {c.block_indices}")
        print(f"    block_types: {[bt.value for bt in c.block_types]}")
        print(f"    overlap: {c.metadata.get('is_continuation', False)}")
        print(f"    text:\n    {c.text[:300]}...")


if __name__ == "__main__":
    # Test DOCX with structure
    test_file(Path("./documents/document_processing_test.docx"), max_tokens=150, overlap_tokens=30)

    # Test DOCX with table
    test_file(Path("./documents/test.docx"), max_tokens=100, overlap_tokens=20)

    # Test TXT (flat)
    test_file(Path("./documents/readme.txt"), max_tokens=100, overlap_tokens=20)

    # Test PDF (page-level paragraphs)
    test_file(Path("./documents/About Aventro Motors.pdf"), max_tokens=200, overlap_tokens=30)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("Good chunks should:")
    print("  - Keep heading + following content together")
    print("  - Not split mid-sentence (unless oversized)")
    print("  - Preserve tables intact (or split with header repeat)")
    print("  - Show heading_path for context")
    print("  - Have deterministic IDs (re-run to verify)")
    print("  - Apply overlap between consecutive chunks")