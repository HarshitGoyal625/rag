"""Hybrid semantic + keyword retriever."""

from __future__ import annotations

from typing import Optional

from embeddings import E5Embedder, VectorStore
from .bm25_retriever import BM25Retriever


class HybridRetriever:
    """Combine semantic vector search with BM25 keyword search."""

    def __init__(
        self,
        embedder: E5Embedder,
        vector_store: VectorStore,
        bm25: BM25Retriever,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
    ):
        if semantic_weight < 0 or keyword_weight < 0:
            raise ValueError(
                "Weights must be non-negative."
            )

        if semantic_weight + keyword_weight == 0:
            raise ValueError(
                "At least one weight must be greater than zero."
            )

        total = semantic_weight + keyword_weight

        self.semantic_weight = (
            semantic_weight / total
        )

        self.keyword_weight = (
            keyword_weight / total
        )

        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25

    def search(
        self,
        query: str,
        k: int = 5,
        candidate_k: int = 20,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """Return top-k results using hybrid retrieval."""

        # ---------------------------------------------------------
        # 1. Semantic retrieval
        # ---------------------------------------------------------

        query_embedding = self.embedder.embed_query(
            query
        )

        semantic_hits = self.vector_store.search(
            query_embedding,
            k=candidate_k,
            filter=filter,
        )

        # ---------------------------------------------------------
        # 2. Keyword retrieval
        # ---------------------------------------------------------

        keyword_hits = self.bm25.search(
            query,
            k=candidate_k,
            filter=filter,
        )

        # ---------------------------------------------------------
        # 3. Normalize scores
        # ---------------------------------------------------------

        semantic_scores = (
            self._normalize_semantic_scores(
                semantic_hits
            )
        )

        keyword_scores = (
            self._normalize_keyword_scores(
                keyword_hits
            )
        )

        # ---------------------------------------------------------
        # 4. Combine candidates
        # ---------------------------------------------------------

        combined = {}

        for hit in semantic_hits:

            chunk_id = hit["id"]

            combined[chunk_id] = {
                "id": chunk_id,
                "text": hit["text"],
                "metadata": hit["metadata"],
                "semantic_score": semantic_scores.get(
                    chunk_id,
                    0.0,
                ),
                "keyword_score": 0.0,
            }

        for hit in keyword_hits:

            chunk_id = hit["id"]

            if chunk_id not in combined:

                combined[chunk_id] = {
                    "id": chunk_id,
                    "text": hit["text"],
                    "metadata": hit["metadata"],
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                }

            combined[chunk_id][
                "keyword_score"
            ] = keyword_scores.get(
                chunk_id,
                0.0,
            )

        # ---------------------------------------------------------
        # 5. Calculate hybrid score
        # ---------------------------------------------------------

        results = []

        for result in combined.values():

            hybrid_score = (
                self.semantic_weight
                * result["semantic_score"]
                +
                self.keyword_weight
                * result["keyword_score"]
            )

            result["hybrid_score"] = hybrid_score

            results.append(result)

        # Highest score first
        results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )

        return results[:k]

    @staticmethod
    def _normalize_semantic_scores(
        hits: list[dict],
    ) -> dict[str, float]:
        """Convert Chroma distance to a 0-1 similarity score."""

        if not hits:
            return {}

        distances = [
            float(hit["distance"])
            for hit in hits
        ]

        min_distance = min(distances)
        max_distance = max(distances)

        if max_distance == min_distance:
            return {
                hit["id"]: 1.0
                for hit in hits
            }

        return {
            hit["id"]: (
                (max_distance - float(hit["distance"]))
                / (max_distance - min_distance)
            )
            for hit in hits
        }

    @staticmethod
    def _normalize_keyword_scores(
        hits: list[dict],
    ) -> dict[str, float]:
        """Normalize BM25 scores to a 0-1 range."""

        if not hits:
            return {}

        scores = [
            float(hit["score"])
            for hit in hits
        ]

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return {
                hit["id"]: 1.0
                for hit in hits
            }

        return {
            hit["id"]: (
                (float(hit["score"]) - min_score)
                / (max_score - min_score)
            )
            for hit in hits
        }