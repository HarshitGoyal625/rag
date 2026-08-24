"""Main entry point for document processing pipeline.

Reads all supported documents in a directory into the structured IR
(`DocumentStructure`), chunks them, embeds them, indexes them,
and demonstrates semantic, keyword, and hybrid retrieval.
"""

from pathlib import Path

from retriever import BM25Retriever, HybridRetriever
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


def chunk_documents(
    documents: dict[str, DocumentStructure],
    **chunker_kwargs
) -> dict[str, list]:
    """Chunk all documents using BlockAwareRecursiveChunker."""
    from document_readers.structure import ChunkerConfig

    # Extract config fields from kwargs
    config_fields = {
        "max_tokens",
        "overlap_tokens",
        "heading_context",
        "table_row_limit",
        "split_long_paragraphs",
        "min_tokens",
    }

    config_kwargs = {
        k: v
        for k, v in chunker_kwargs.items()
        if k in config_fields
    }

    config = (
        ChunkerConfig(**config_kwargs)
        if config_kwargs
        else ChunkerConfig()
    )

    chunker = BlockAwareRecursiveChunker(config=config)
    all_chunks: dict[str, list] = {}

    for path, doc in documents.items():
        chunks = chunker.chunk(doc)
        all_chunks[path] = chunks
        print(f"✓ Chunked: {path} -> {len(chunks)} chunks")

    return all_chunks


def flatten_chunks(all_chunks: dict[str, list]) -> list:
    """Flatten document-wise chunks into a single list."""
    flat_chunks = []

    for chunks in all_chunks.values():
        flat_chunks.extend(chunks)

    return flat_chunks


def embed_and_index(
    all_chunks: dict[str, list],
    embedder: E5Embedder,
    store: VectorStore,
) -> None:
    """Embed all chunks and upsert into vector store."""

    print("\n" + "=" * 50)
    print("EMBEDDING & INDEXING")
    print("=" * 50)

    total_chunks = sum(
        len(chunks)
        for chunks in all_chunks.values()
    )

    print(
        f"Embedding {total_chunks} chunks with "
        f"{embedder.model_name} "
        f"(dim={embedder.dimension})..."
    )

    # Flatten all chunks
    flat_chunks = flatten_chunks(all_chunks)

    # Embed in batches
    texts = [chunk.text for chunk in flat_chunks]

    embeddings = embedder.embed(texts)

    # Upsert to vector store
    store.upsert_chunks(
        flat_chunks,
        embeddings,
    )

    print(f"✓ Indexed {len(flat_chunks)} chunks")


def search_demo(
    store: VectorStore,
    embedder: E5Embedder,
    queries: list[str],
) -> None:
    """Run semantic/vector search queries and display results."""

    print("\n" + "=" * 50)
    print("SEMANTIC SEARCH")
    print("=" * 50)

    for query in queries:
        print(f"\n🔍 Query: {query}")

        query_embedding = embedder.embed_query(query)

        hits = store.search(
            query_embedding,
            k=3,
        )

        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]

            src = Path(meta["source"]).name

            pages = (
                f"pp.{meta['page_start']}-{meta['page_end']}"
                if meta["page_start"] != -1
                else "N/A"
            )

            heading = (
                meta["heading_path"]
                if meta["heading_path"]
                else "(no heading)"
            )

            print(
                f"  {i}. [{src} {pages}] {heading}"
            )

            print(
                f"     distance={hit['distance']:.3f} | "
                f"{hit['text'][:120]}..."
            )


def keyword_search_demo(
    bm25: BM25Retriever,
    queries: list[str],
) -> None:
    """Run BM25 keyword searches and display results."""

    print("\n" + "=" * 50)
    print("BM25 KEYWORD SEARCH")
    print("=" * 50)

    for query in queries:
        print(f"\n🔎 Query: {query}")

        hits = bm25.search(
            query,
            k=3,
        )

        if not hits:
            print("  No results found.")
            continue

        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]

            src = Path(meta["source"]).name

            page_range = meta.get("page_range")

            pages = (
                f"pp.{page_range[0]}-{page_range[1]}"
                if page_range
                else "N/A"
            )

            heading_path = meta.get(
                "heading_path",
                [],
            )

            heading = (
                " > ".join(heading_path)
                if heading_path
                else "(no heading)"
            )

            print(
                f"  {i}. [{src} {pages}] {heading}"
            )

            print(
                f"     BM25 score={hit['score']:.3f} | "
                f"{hit['text'][:120]}..."
            )


def hybrid_search_demo(
    hybrid: HybridRetriever,
    queries: list[str],
) -> None:
    """Run hybrid semantic + keyword searches."""

    print("\n" + "=" * 50)
    print("HYBRID SEARCH")
    print("=" * 50)

    print(
        f"Semantic weight: "
        f"{hybrid.semantic_weight:.2f}"
    )

    print(
        f"Keyword weight: "
        f"{hybrid.keyword_weight:.2f}"
    )

    for query in queries:
        print(f"\n🔎 Query: {query}")

        hits = hybrid.search(
            query,
            k=3,
            candidate_k=20,
        )

        if not hits:
            print("  No results found.")
            continue

        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]

            src = Path(meta["source"]).name

            # Handle metadata coming from either
            # ChromaDB or BM25.
            if "page_start" in meta:
                pages = (
                    f"pp.{meta['page_start']}-{meta['page_end']}"
                    if meta["page_start"] != -1
                    else "N/A"
                )
            else:
                page_range = meta.get("page_range")

                pages = (
                    f"pp.{page_range[0]}-{page_range[1]}"
                    if page_range
                    else "N/A"
                )

            heading = meta.get(
                "heading_path",
                "(no heading)",
            )

            if isinstance(heading, list):
                heading = (
                    " > ".join(heading)
                    if heading
                    else "(no heading)"
                )

            print(
                f"  {i}. [{src} {pages}] {heading}"
            )

            print(
                f"     hybrid={hit['hybrid_score']:.3f} | "
                f"semantic={hit['semantic_score']:.3f} | "
                f"keyword={hit['keyword_score']:.3f}"
            )

            print(
                f"     {hit['text'][:120]}..."
            )


if __name__ == "__main__":

    # ============================================================
    # DOCUMENT READING
    # ============================================================

    docs_dir = "./documents"

    print(
        f"Reading documents from: {docs_dir}"
    )

    print(
        f"Supported extensions: "
        f"{DocumentReaderFactory.get_supported_extensions()}"
    )

    print("-" * 50)

    documents = read_documents(docs_dir)

    print("-" * 50)

    print(
        f"Total documents read: "
        f"{len(documents)}"
    )

    for path, doc in documents.items():

        head_count = len(
            doc.iter_headings()
        )

        print(
            f"\n--- {path} --- "
            f"doc_type={doc.doc_type} "
            f"blocks={len(doc.blocks)} "
            f"headings={head_count}"
        )

        for block in doc.blocks[:8]:

            lvl = (
                f" (L{block.level})"
                if block.level is not None
                else ""
            )

            pg = (
                f" [p{block.page}]"
                if block.page is not None
                else ""
            )

            print(
                f"  [{block.type.value}{lvl}{pg}] "
                f"{block.text[:70]}"
            )

        if len(doc.blocks) > 8:
            print(
                f"  ... "
                f"({len(doc.blocks) - 8} more blocks)"
            )

    # ============================================================
    # CHUNKING
    # ============================================================

    print("\n" + "=" * 50)

    print(
        "CHUNKING "
        "(block_aware_recursive, max_tokens=512)"
    )

    print("=" * 50)

    all_chunks = chunk_documents(
        documents,
        max_tokens=512,
        overlap_tokens=50,
    )

    print("\n" + "-" * 50)
    print("CHUNK SUMMARIES")
    print("-" * 50)

    total_chunks = 0

    for path, chunks in all_chunks.items():

        total_chunks += len(chunks)

        print(
            f"\n--- {path} --- "
            f"{len(chunks)} chunks"
        )

        for chunk in chunks[:3]:

            pg = (
                f" p{chunk.page_range[0]}-"
                f"{chunk.page_range[1]}"
                if chunk.page_range
                else ""
            )

            hp = (
                f" | headings: "
                f"{chunk.heading_path}"
                if chunk.heading_path
                else ""
            )

            print(
                f"  [{chunk.token_count} tok"
                f"{pg}{hp}] "
                f"blocks {chunk.block_indices}: "
                f"{chunk.text[:80]}..."
            )

        if len(chunks) > 3:
            print(
                f"  ... "
                f"({len(chunks) - 3} more chunks)"
            )

    print(
        f"\nTotal chunks: {total_chunks}"
    )

    # ============================================================
    # PREPARE RETRIEVAL DATA
    # ============================================================

    flat_chunks = flatten_chunks(
        all_chunks
    )

    print(
        f"\n✓ Prepared {len(flat_chunks)} "
        f"chunks for retrieval"
    )

    # ============================================================
    # BM25 KEYWORD INDEX
    # ============================================================

    bm25 = BM25Retriever(
        flat_chunks
    )

    print(
        "✓ BM25 keyword index created"
    )

    # ============================================================
    # EMBEDDING & VECTOR INDEX
    # ============================================================

    embedder = E5Embedder(
        model_name="intfloat/e5-small-v2"
    )

    store = VectorStore(
        path="./chroma_db",
        collection_name="rag_chunks",
    )

    embed_and_index(
        all_chunks,
        embedder,
        store,
    )

    # ============================================================
    # HYBRID RETRIEVER
    # ============================================================

    hybrid = HybridRetriever(
        embedder=embedder,
        vector_store=store,
        bm25=bm25,
        semantic_weight=0.5,
        keyword_weight=0.5,
    )

    print(
        "✓ Hybrid retriever created"
    )

    # ============================================================
    # TEST QUERIES
    # ============================================================

    queries = [
        "What is Aventro Motors company overview?",
        "Who are the founders of Aventro?",
        "Market risk interest rate risk",
        "Document processing system steps",
    ]

    # ============================================================
    # SEMANTIC SEARCH
    # ============================================================

    search_demo(
        store,
        embedder,
        queries,
    )

    # ============================================================
    # BM25 KEYWORD SEARCH
    # ============================================================

    keyword_search_demo(
        bm25,
        queries,
    )

    # ============================================================
    # HYBRID SEARCH
    # ============================================================

    hybrid_search_demo(
        hybrid,
        queries,
    )

    # ============================================================
    # FINAL VECTOR COUNT
    # ============================================================

    print(
        f"\n✓ Total vectors in store: "
        f"{store.count()}"
    )