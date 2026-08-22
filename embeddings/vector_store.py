"""ChromaDB vector store for chunk embeddings."""
from __future__ import annotations

from typing import Optional

import chromadb
from chromadb.config import Settings

from document_readers.structure import Chunk


class VectorStore:
    """Persistent vector store using ChromaDB."""

    def __init__(
        self,
        path: str = "./chroma_db",
        collection_name: str = "chunks",
    ):
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks with their embeddings. Uses deterministic chunk_id as ID."""
        if not chunks:
            return

        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[self._chunk_metadata(c) for c in chunks],
        )

    def _chunk_metadata(self, chunk: Chunk) -> dict:
        page_range = chunk.metadata.get("page_range", [])
        return {
            "source": chunk.metadata.get("source", ""),
            "page_start": page_range[0] if page_range else -1,
            "page_end": page_range[-1] if page_range else -1,
            "file_type": chunk.metadata.get("file_type", ""),
            "heading_path": " > ".join(chunk.metadata.get("heading_path", [])),
            "block_types": ",".join(chunk.metadata.get("block_types", [])),
            "chunk_index": chunk.metadata.get("chunk_index", -1),
        }

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """Search for similar chunks. Returns list of dicts with text, metadata, distance."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter,
            include=["documents", "metadatas", "distances"],
        )

        # Flatten Chroma's columnar results
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return hits

    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a specific source file."""
        self.collection.delete(where={"source": source})

    def count(self) -> int:
        return self.collection.count()

    def get_by_ids(self, ids: list[str]) -> list[dict]:
        """Fetch chunks by their IDs."""
        results = self.collection.get(ids=ids, include=["documents", "metadatas"])
        return [
            {"id": results["ids"][i], "text": results["documents"][i], "metadata": results["metadatas"][i]}
            for i in range(len(results["ids"]))
        ]