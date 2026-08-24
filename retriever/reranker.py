"""Cross-encoder reranker for retrieved RAG chunks."""

from __future__ import annotations

from typing import Any, Optional


class CrossEncoderReranker:
    """Rerank retrieved chunks using a sentence-transformers cross encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(
            model_name,
            device=device,
        )

    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank retrieved hits and return the best candidates."""

        if not hits:
            return []

        pairs = [
            (query, hit["text"])
            for hit in hits
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        reranked = []

        for hit, score in zip(hits, scores):
            result = dict(hit)

            result["rerank_score"] = float(score)

            reranked.append(result)

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]