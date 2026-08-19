from document_readers.factory import DocumentReaderFactory
from ingestion.chunker import TextChunkers
def main():
    service = TextChunkers(
        chunk_size=2000, chunk_overlap=300
    )
    chunks = service.ingest("./data")
    print("\n == Sample Chunks == \n")
    for chunk in chunks[:5]:
        print(
            f"ID : {chunk['id']} "
        )
        print(f"Source : {chunk['source']} ")
        print(f"Page : {chunk['Page']} ")
        print(f"File Type : {chunk['file_type']} ")
        print (f"Chunk Index : {chunk['chunk_index']} ")

        print(f"Text : {chunk['text'][:500]}")
        print("\n--------------------\n")
if __name__ == "__main__":
    main()
