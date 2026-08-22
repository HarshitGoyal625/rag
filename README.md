# Custom RAG Pipeline

A modular document processing pipeline for Retrieval-Augmented Generation: **read → structure → chunk → embed → retrieve**.

## What It Does

| Stage | Purpose |
|-------|---------|
| **Read** | Extract structured content from PDF, DOCX, TXT/MD preserving headings, tables, lists, page boundaries |
| **Structure** | Normalize all formats into a typed intermediate representation (`DocumentStructure`) — chunker never sees file type |
| **Chunk** | Block-aware recursive chunking that respects document hierarchy (headings drive boundaries), splits oversized blocks by sentence/table rows, applies token overlap, emits deterministic IDs |
| **Embed** | Local embeddings via sentence-transformers (E5 family) or OpenAI-compatible API |
| **Index** | Persistent vector store (ChromaDB) with metadata filtering (source, file type, page range) |
| **Retrieve** | Semantic search with citations (page range, heading path) |
