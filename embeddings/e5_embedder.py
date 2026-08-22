"""E5 embedder implementation using sentence-transformers."""
from __future__ import annotations

from typing import Optional

from .base import Embedder


class E5Embedder:
    """E5-family embedder (intfloat/e5-*) with passage/query prefixes."""

    def __init__(
        self,
        model_name: str = "intfloat/e5-small-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        self._model = SentenceTransformer(model_name, device=device)
        # Use get_embedding_dimension (newer API), fallback for older versions
        self._dimension = getattr(self._model, 'get_embedding_dimension',
                                   lambda: self._model.get_sentence_embedding_dimension())()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed passages with 'passage: ' prefix."""
        passages = [f"passage: {t}" for t in texts]
        embeddings = self._model.encode(
            passages,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed query with 'query: ' prefix."""
        embedding = self._model.encode(
            f"query: {query}",
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embedding.tolist()