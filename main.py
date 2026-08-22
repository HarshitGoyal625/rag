"""Main entry point for document processing pipeline.

Reads all supported documents in a directory into the structured IR
(`DocumentStructure`), chunks them, embeds them, indexes them,
and demonstrates retrieval.
"""
from pathlib import Path

from document_readers import DocumentReaderFactory
from document_readers import DocumentStructure
from chunkers import BlockAwareRecursiveChunker
from embeddings import E5Embedder, VectorStore


def read_documents(directory: str) -> dict[str, DocumentStructure]:
    """
    Recursively read all supported documents in a directory.

    Args:
        directory: Path to the documents directory

    Returns:
        Dictionary mapping file paths to their structured IR
        (`DocumentStructure`).
    """
    documents: dict[str, DocumentStructure] = {}
    root_path = Path(directory)

    for file_path in root_path.rglob("*"):
        if file_path.is_file():
            try:
                doc = DocumentReaderFactory.read_document(file_path)
                documents[str(file_path)] = doc
                print(f"✓ Read: {file_path}")
            except ValueError as e:
                print(f"✗ Skipped {file_path}: {e}")

    return documents


def chunk_documents(documents: dict[str, DocumentStructure], **chunker_kwargs) -> dict[str, list]:
    """Chunk all documents using BlockAwareRecursiveChunker."""
    from document_readers.structure import ChunkerConfig
    
    # Extract config fields from kwargs
    config_fields = {"max_tokens", "overlap_tokens", "heading_context",
                     "table_row_limit", "split_long_paragraphs", "min_tokens"}
    config_kwargs = {k: v for k, v in chunker_kwargs.items() if k in config_fields}
    config = ChunkerConfig(**config_kwargs) if config_kwargs else ChunkerConfig()
    
    chunker = BlockAwareRecursiveChunker(config=config)
    all_chunks: dict[str, list] = {}

    for path, doc in documents.items():
        chunks = chunker.chunk(doc)
        all_chunks[path] = chunks
        print(f"✓ Chunked: {path} -> {len(chunks)} chunks")

    return all_chunks


def embed_and_index(all_chunks: dict[str, list], embedder: E5Embedder, store: VectorStore) -> None:
    """Embed all chunks and upsert into vector store."""
    print("\n" + "=" * 50)
    print("EMBEDDING & INDEXING")
    print("=" * 50)

    total_chunks = sum(len(c) for c in all_chunks.values())
    print(f"Embedding {total_chunks} chunks with {embedder.model_name} (dim={embedder.dimension})...")

    # Flatten all chunks
    flat_chunks = []
    for chunks in all_chunks.values():
        flat_chunks.extend(chunks)

    # Embed in batches
    texts = [c.text for c in flat_chunks]
    embeddings = embedder.embed(texts)

    # Upsert to vector store
    store.upsert_chunks(flat_chunks, embeddings)
    print(f"✓ Indexed {len(flat_chunks)} chunks")


def search_demo(store: VectorStore, embedder: E5Embedder, queries: list[str]) -> None:
    """Run search queries and display results."""
    print("\n" + "=" * 50)
    print("SEARCH DEMO")
    print("=" * 50)

    for query in queries:
        print(f"\n🔍 Query: {query}")
        query_emb = embedder.embed_query(query)
        hits = store.search(query_emb, k=3)

        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]
            src = Path(meta["source"]).name
            pages = f"pp.{meta['page_start']}-{meta['page_end']}" if meta["page_start"] != -1 else "N/A"
            heading = meta["heading_path"] or "(no heading)"
            dist = hit["distance"]
            print(f"  {i}. [{src} {pages}] {heading}")
            print(f"     dist={dist:.3f} | {hit['text'][:120]}...")


if __name__ == "__main__":
    docs_dir = "./documents"
    print(f"Reading documents from: {docs_dir}")
    print(f"Supported extensions: {DocumentReaderFactory.get_supported_extensions()}")
    print("-" * 50)

    documents = read_documents(docs_dir)

    print("-" * 50)
    print(f"Total documents read: {len(documents)}")
    for path, doc in documents.items():
        head_count = len(doc.iter_headings())
        print(
            f"\n--- {path} --- "
            f"doc_type={doc.doc_type} blocks={len(doc.blocks)} headings={head_count}"
        )
        for block in doc.blocks[:8]:
            lvl = f" (L{block.level})" if block.level is not None else ""
            pg = f" [p{block.page}]" if block.page is not None else ""
            print(f"  [{block.type.value}{lvl}{pg}] {block.text[:70]}")
        if len(doc.blocks) > 8:
            print(f"  ... ({len(doc.blocks) - 8} more blocks)")

    print("\n" + "=" * 50)
    print("CHUNKING (block_aware_recursive, max_tokens=512)")
    print("=" * 50)

    all_chunks = chunk_documents(documents, max_tokens=512, overlap_tokens=50)

    print("\n" + "-" * 50)
    print("CHUNK SUMMARIES")
    print("-" * 50)
    total_chunks = 0
    for path, chunks in all_chunks.items():
        total_chunks += len(chunks)
        print(f"\n--- {path} --- {len(chunks)} chunks")
        for c in chunks[:3]:
            pg = f" p{c.page_range[0]}-{c.page_range[1]}" if c.page_range else ""
            hp = f" | headings: {c.heading_path}" if c.heading_path else ""
            print(f"  [{c.token_count} tok{pg}{hp}] blocks {c.block_indices}: {c.text[:80]}...")
        if len(chunks) > 3:
            print(f"  ... ({len(chunks) - 3} more chunks)")

    print(f"\nTotal chunks: {total_chunks}")

    # --- Embedding & Indexing ---
    embedder = E5Embedder(model_name="intfloat/e5-small-v2")
    store = VectorStore(path="./chroma_db", collection_name="rag_chunks")

    embed_and_index(all_chunks, embedder, store)

    # --- Search Demo ---
    queries = [
        "What is Aventro Motors company overview?",
        "Who are the founders of Aventro?",
        "Market risk interest rate risk",
        "Document processing system steps",
    ]
    search_demo(store, embedder, queries)

    print(f"\n✓ Total vectors in store: {store.count()}")