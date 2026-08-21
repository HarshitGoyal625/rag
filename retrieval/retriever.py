from .embeddings import E5embedder
from .vector_store import VectorStore
class Retriever:
    def __init__(self):
        self.embedder = E5embedder()
        self.vector_store = VectorStore()
    def retrieve(
            self,
            query: str,
            top_k: int = 5
    ):
        query_embedding = (
            self.embedder.embed_query(query)
        )
        return self.vector_store.search(
            query_embedding,
            top_k
        )