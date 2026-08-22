from sentence_transformers import SentenceTransformer
class E5embedder :
    def __init__(
            self,
            model_name: str = "intfloat/e5-small-v2"
    ):
        self.model = SentenceTransformer(model_name)
    def embed_docs(
        self,
        documents :list[str]
    ) :
        passages = [
            f"passage : {document}"
            for document in documents
        ]
        return self.model.encode(
            passages,
            normalize_embeddings= True,
            show_progress_bar= True
        )
    def embed_query(self, query: str):
        return self.model.encode(
            f"query : {query}",
            normalize_embeddings= True
        )
        