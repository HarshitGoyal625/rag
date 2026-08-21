from ingestion import ingestion_service #Need to be changed acc to the changes and merging the service into document_readers.
from retrieval import E5embedder, VectorStore
def main():
    print("\n----Ingestion----\n")
    ingestion_service = ingestion_service()
    chunks = ingestion_service.ingest(
        "./data"
    )
    print(
        f"Total chunks: {len(chunks)}"
    )
    #Embeddings
    embedder = E5embedder()
    texts = [
        chunk["text"]
        for chunk in chunks 
    ]
    embeddings = embedder.embed_documents(
        texts
    )
    print(f"{embeddings.shape}")

    vector_store = VectorStore()
    vector_store.add_documents(
        chunks,
        embeddings
    )
    print(
        f"Vector Stored"
        f"{vector_store.count()}"
    )
if __name__ == "__main__":
    main()

