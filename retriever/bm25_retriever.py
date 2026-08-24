"""BM25 keyword retriever."""

from __future__ import annotations

import re
from typing import Optional

from document_readers.structure import Chunk


class BM25Retriever:
    """Keyword-based retriever using BM25."""

    def __init__(
        self,
        chunks: list[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self._tokenized_corpus = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]

        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=k1,
            b=b,
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple normalized word tokenizer."""
        return re.findall(r"\b\w+\b", text.lower())

    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """Search chunks using BM25 keyword relevance."""

        query_tokens = self._tokenize(query)

        scores = self._bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        hits = []

        for idx in ranked_indices:
            if len(hits) >= k:
                break

            chunk = self.chunks[idx]

            if not self._matches_filter(chunk, filter):
                continue

            hits.append({
                "id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": {
                    "source": chunk.source,
                    "file_type": chunk.doc_type,
                    "page_range": chunk.page_range,
                    "heading_path": chunk.heading_path,
                    "block_types": [
                        bt.value for bt in chunk.block_types
                    ],
                    "chunk_index": chunk.metadata.get(
                        "chunk_index",
                        -1,
                    ),
                },
                "score": float(scores[idx]),
            })

        return hits

    @staticmethod
    def _matches_filter(
        chunk: Chunk,
        filter: Optional[dict],
    ) -> bool:
        """Apply simple metadata filtering."""

        if not filter:
            return True

        for key, expected_value in filter.items():

            if key == "source":
                actual_value = chunk.source

            elif key == "file_type":
                actual_value = chunk.doc_type

            else:
                actual_value = chunk.metadata.get(key)

            if actual_value != expected_value:
                return False

        return True