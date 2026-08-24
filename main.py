"""Main entry point for document processing pipeline.

Reads all supported documents in a directory into the structured IR
(`DocumentStructure`), chunks them, embeds them, indexes them,
and demonstrates semantic, keyword, hybrid, and reranked retrieval.
"""

from pathlib import Path

from retriever import (
    BM25Retriever,
    HybridRetriever,
    CrossEncoderReranker,
)

from document_readers import DocumentReaderFactory
from document_readers import DocumentStructure

from chunkers import BlockAwareRecursiveChunker

from embeddings import E5Embedder, VectorStore


# ============================================================
# DOCUMENT READING
# ============================================================

def read_documents(
    directory: str,
) -> dict[str, DocumentStructure]:
    """
    Recursively read all supported documents in a directory.

    Args:
        directory: Path to the documents directory

    Returns:
        Dictionary mapping file paths to their structured IR.
    """

    documents: dict[str, DocumentStructure] = {}

    root_path = Path(directory)

    for file_path in root_path.rglob("*"):

        if file_path.is_file():

            try:
                doc = DocumentReaderFactory.read_document(
                    file_path
                )

                documents[str(file_path)] = doc

                print(f"✓ Read: {file_path}")

            except ValueError as e:

                print(
                    f"✗ Skipped {file_path}: {e}"
                )

    return documents


# ============================================================
# CHUNKING
# ============================================================

def chunk_documents(
    documents: dict[str, DocumentStructure],
    **chunker_kwargs,
) -> dict[str, list]:
    """Chunk all documents using BlockAwareRecursiveChunker."""

    from document_readers.structure import ChunkerConfig

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

    chunker = BlockAwareRecursiveChunker(
        config=config
    )

    all_chunks: dict[str, list] = {}

    for path, doc in documents.items():

        chunks = chunker.chunk(doc)

        all_chunks[path] = chunks

        print(
            f"✓ Chunked: {path} -> "
            f"{len(chunks)} chunks"
        )

    return all_chunks


def flatten_chunks(
    all_chunks: dict[str, list],
) -> list:
    """Flatten document-wise chunks into a single list."""

    flat_chunks = []

    for chunks in all_chunks.values():
        flat_chunks.extend(chunks)

    return flat_chunks


# ============================================================
# EMBEDDING + VECTOR INDEX
# ============================================================

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

    flat_chunks = flatten_chunks(
        all_chunks
    )

    texts = [
        chunk.text
        for chunk in flat_chunks
    ]

    embeddings = embedder.embed(
        texts
    )

    store.upsert_chunks(
        flat_chunks,
        embeddings,
    )

    print(
        f"✓ Indexed {len(flat_chunks)} chunks"
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def _get_source_and_pages(
    metadata: dict,
) -> tuple[str, str]:
    """Extract source filename and page information."""

    src = Path(
        metadata.get("source", "")
    ).name

    # ChromaDB metadata
    if "page_start" in metadata:

        pages = (
            f"pp.{metadata['page_start']}-"
            f"{metadata['page_end']}"
            if metadata["page_start"] != -1
            else "N/A"
        )

        return src, pages

    # BM25 metadata
    page_range = metadata.get(
        "page_range"
    )

    pages = (
        f"pp.{page_range[0]}-"
        f"{page_range[1]}"
        if page_range
        else "N/A"
    )

    return src, pages


def _get_heading(
    metadata: dict,
) -> str:
    """Extract heading path from metadata."""

    heading = metadata.get(
        "heading_path",
        [],
    )

    if isinstance(heading, list):

        return (
            " > ".join(heading)
            if heading
            else "(no heading)"
        )

    return (
        heading
        if heading
        else "(no heading)"
    )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def search_demo(
    store: VectorStore,
    embedder: E5Embedder,
    queries: list[str],
) -> None:
    """Run semantic/vector search queries."""

    print("\n" + "=" * 50)
    print("SEMANTIC SEARCH")
    print("=" * 50)

    for query in queries:

        print(
            f"\n🔍 Query: {query}"
        )

        query_embedding = (
            embedder.embed_query(query)
        )

        hits = store.search(
            query_embedding,
            k=3,
        )

        if not hits:

            print(
                "  No results found."
            )

            continue

        for i, hit in enumerate(
            hits,
            1,
        ):

            meta = hit["metadata"]

            src, pages = (
                _get_source_and_pages(meta)
            )

            heading = _get_heading(
                meta
            )

            print(
                f"  {i}. "
                f"[{src} {pages}] "
                f"{heading}"
            )

            print(
                f"     distance="
                f"{hit['distance']:.3f} | "
                f"{hit['text'][:120]}..."
            )


# ============================================================
# BM25 KEYWORD SEARCH
# ============================================================

def keyword_search_demo(
    bm25: BM25Retriever,
    queries: list[str],
) -> None:
    """Run BM25 keyword searches."""

    print("\n" + "=" * 50)
    print("BM25 KEYWORD SEARCH")
    print("=" * 50)

    for query in queries:

        print(
            f"\n🔎 Query: {query}"
        )

        hits = bm25.search(
            query,
            k=3,
        )

        if not hits:

            print(
                "  No results found."
            )

            continue

        for i, hit in enumerate(
            hits,
            1,
        ):

            meta = hit["metadata"]

            src, pages = (
                _get_source_and_pages(meta)
            )

            heading = _get_heading(
                meta
            )

            print(
                f"  {i}. "
                f"[{src} {pages}] "
                f"{heading}"
            )

            print(
                f"     BM25 score="
                f"{hit['score']:.3f} | "
                f"{hit['text'][:120]}..."
            )


# ============================================================
# HYBRID SEARCH
# ============================================================

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

        print(
            f"\n🔎 Query: {query}"
        )

        hits = hybrid.search(
            query,
            k=3,
            candidate_k=20,
        )

        if not hits:

            print(
                "  No results found."
            )

            continue

        for i, hit in enumerate(
            hits,
            1,
        ):

            meta = hit["metadata"]

            src, pages = (
                _get_source_and_pages(meta)
            )

            heading = _get_heading(
                meta
            )

            print(
                f"  {i}. "
                f"[{src} {pages}] "
                f"{heading}"
            )

            print(
                f"     hybrid="
                f"{hit['hybrid_score']:.3f} | "
                f"semantic="
                f"{hit['semantic_score']:.3f} | "
                f"keyword="
                f"{hit['keyword_score']:.3f}"
            )

            print(
                f"     {hit['text'][:120]}..."
            )


# ============================================================
# RERANKED HYBRID SEARCH
# ============================================================

def reranked_search_demo(
    hybrid: HybridRetriever,
    reranker: CrossEncoderReranker,
    queries: list[str],
) -> None:
    """
    Run two-stage retrieval.

    Stage 1:
        Hybrid retrieval generates a candidate pool.

    Stage 2:
        Cross-encoder reranks those candidates.
    """

    print("\n" + "=" * 50)
    print("RERANKED HYBRID SEARCH")
    print("=" * 50)

    print(
        "Stage 1: Hybrid retrieval"
    )

    print(
        "Stage 2: Cross-encoder reranking"
    )

    print(
        "Candidates: 20"
    )

    print(
        "Final results: 5"
    )

    for query in queries:

        print(
            f"\n🔎 Query: {query}"
        )

        # ----------------------------------------------------
        # Stage 1: Candidate retrieval
        # ----------------------------------------------------

        candidates = hybrid.search(
            query,
            k=20,
            candidate_k=20,
        )

        if not candidates:

            print(
                "  No candidates found."
            )

            continue

        # ----------------------------------------------------
        # Stage 2: Reranking
        # ----------------------------------------------------

        hits = reranker.rerank(
            query=query,
            hits=candidates,
            top_k=5,
        )

        if not hits:

            print(
                "  No reranked results found."
            )

            continue

        # ----------------------------------------------------
        # Display final results
        # ----------------------------------------------------

        for i, hit in enumerate(
            hits,
            1,
        ):

            meta = hit["metadata"]

            src, pages = (
                _get_source_and_pages(meta)
            )

            heading = _get_heading(
                meta
            )

            print(
                f"  {i}. "
                f"[{src} {pages}] "
                f"{heading}"
            )

            print(
                f"     rerank="
                f"{hit['rerank_score']:.4f} | "
                f"hybrid="
                f"{hit.get('hybrid_score', 0.0):.3f}"
            )

            print(
                f"     {hit['text'][:200]}..."
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # DOCUMENT READING
    # ========================================================

    docs_dir = "./documents"

    print(
        f"Reading documents from: "
        f"{docs_dir}"
    )

    print(
        f"Supported extensions: "
        f"{DocumentReaderFactory.get_supported_extensions()}"
    )

    print("-" * 50)

    documents = read_documents(
        docs_dir
    )

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
                f"  [{block.type.value}"
                f"{lvl}{pg}] "
                f"{block.text[:70]}"
            )

        if len(doc.blocks) > 8:

            print(
                f"  ... "
                f"({len(doc.blocks) - 8} more blocks)"
            )

    # ========================================================
    # CHUNKING
    # ========================================================

    print(
        "\n" + "=" * 50
    )

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

    print(
        "\n" + "-" * 50
    )

    print(
        "CHUNK SUMMARIES"
    )

    print(
        "-" * 50
    )

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
        f"\nTotal chunks: "
        f"{total_chunks}"
    )

    # ========================================================
    # PREPARE RETRIEVAL DATA
    # ========================================================

    flat_chunks = flatten_chunks(
        all_chunks
    )

    print(
        f"\n✓ Prepared "
        f"{len(flat_chunks)} "
        f"chunks for retrieval"
    )

    # ========================================================
    # BM25 KEYWORD INDEX
    # ========================================================

    bm25 = BM25Retriever(
        flat_chunks
    )

    print(
        "✓ BM25 keyword index created"
    )

    # ========================================================
    # EMBEDDING + VECTOR INDEX
    # ========================================================

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

    # ========================================================
    # HYBRID RETRIEVER
    # ========================================================

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

    # ========================================================
    # CROSS-ENCODER RERANKER
    # ========================================================

    reranker = CrossEncoderReranker(
        model_name=(
            "cross-encoder/"
            "ms-marco-MiniLM-L-6-v2"
        )
    )

    print(
        "✓ Cross-encoder reranker created"
    )

    # ========================================================
    # TEST QUERIES
    # ========================================================

    queries = [
        "What is Aventro Motors company overview?",
        "Who are the founders of Aventro?",
        "Market risk interest rate risk",
        "Document processing system steps",
    ]

    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    search_demo(
        store,
        embedder,
        queries,
    )

    # ========================================================
    # BM25 KEYWORD SEARCH
    # ========================================================

    keyword_search_demo(
        bm25,
        queries,
    )

    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    hybrid_search_demo(
        hybrid,
        queries,
    )

    # ========================================================
    # RERANKED HYBRID SEARCH
    # ========================================================

    reranked_search_demo(
        hybrid,
        reranker,
        queries,
    )

    # ========================================================
    # FINAL VECTOR COUNT
    # ========================================================

    print(
        f"\n✓ Total vectors in store: "
        f"{store.count()}"
    )