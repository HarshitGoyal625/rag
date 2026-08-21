import chromadb
class VectorStore:
    def __init__(
            self,
            path : str = "./chroma_db",
            collection_name: str = "Documents"
    ): 
        self.client = chromadb.PersistentClient(
            path = path
        )
        self.collection = (
            self.client.get_or_create_collection(
                name = collection_name,
                configuration = {
                    "hnsw" : {
                        "space" : "cosine"
                    }
                }
            )
        )
    def add_docs(
            self,
            chunks: list[dict],
            embeddings
    ):
        ids = [
            chunk["text"]
            for chunk in chunks
        ]
        documents = [
            chunk["text"]
            for chunk in chunks
        ]
        metadata = [
            {
                "source": chunk["source"],
                "page":(
                  chunk["page"] 
                  if chunk["page"] is not None
                  else -1
                ),
                "file_type": chunk["file_type"],
                "chunk_index": chunk["chunk_index"]
            }
            for chunk in chunks
        ]
        self.collection.upsert(
            ids = ids,
            documents= documents,
            embeddings= embeddings.tolist(),
            metadatas= metadata
        )
    def search(
            self,
            query_embedding,
            top_k: int=5 #we can change it later
    ) :
        return self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results = top_k
        )
    def count(self):
        return self.collection.count()