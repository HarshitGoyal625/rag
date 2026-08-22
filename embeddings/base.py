"""Embedder protocol and base types."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Protocol for text embedding models."""

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents/passages."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query (may use different prefix/instructions)."""
        ...