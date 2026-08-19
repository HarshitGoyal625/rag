from typing import List
class TextChunker :
    def __init__ (self, chunk_size: int = 2000, chunk_overlap: int = 300) :
        if chunk_overlap >= chunk_size :
            raise ValueError ("Chunk_overlap must be smaller than the chunk size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        def chunk_text(self, text: str) -> List[str]:
            paragraphs = [paragraph.strip()
                          for paragraph in text.split("\n")
                          if paragraph.strip()]
            chunks = []
            current_chunk = ""
            for paragraph in paragraphs:
                candidate = (
                    f"{current_chunk}/n{paragraph}"
                    if current_chunk 
                    else paragraph
                )
                if len(candidate) <= self.chunk_size:
                    current_chunk = candidate
                if current_chunk:
                    chunks.append(current_chunk.strip())

                current_chunk = paragraph

            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks
        def chunk_documents(self, documents: List[dict]) -> List[dict]:
            chunks = []
            global_chunk_index = 0
            for document in documents:
                text_chunks = self.chunk_text(document["text"])
                for chunk_index, chunk in enumerate(text_chunks):
                    chunks.append(
                        {
                            "id": f"chunk_{global_chunk_index}",
                            "text": chunk,
                            "source": document["source"],
                            "page": document.get("page"),
                            "file_type": document["file_type"],
                            "chunk_index": chunk_index,
                        }
                    )
                    global_chunk_index += 1
            return chunks
