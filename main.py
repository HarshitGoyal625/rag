"""Main entry point for document processing pipeline."""
from pathlib import Path
from document_readers import DocumentReaderFactory


def read_documents(directory: str) -> dict[str, str]:
    """
    Recursively read all supported documents in a directory.

    Args:
        directory: Path to the documents directory

    Returns:
        Dictionary mapping file paths to their extracted text content
    """
    documents = {}
    root_path = Path(directory)

    for file_path in root_path.rglob('*'):
        if file_path.is_file():
            try:
                text = DocumentReaderFactory.read_document(file_path)
                documents[str(file_path)] = text
                print(f"✓ Read: {file_path}")
            except ValueError as e:
                print(f"✗ Skipped {file_path}: {e}")

    return documents


if __name__ == "__main__":
    docs_dir = "./documents"
    print(f"Reading documents from: {docs_dir}")
    print(f"Supported extensions: {DocumentReaderFactory.get_supported_extensions()}")
    print("-" * 50)

    documents = read_documents(docs_dir)

    print("-" * 50)
    print(f"Total documents read: {len(documents)}")
    for path, content in documents.items():
        print(f"\n--- {path} ---")
        print(content[:500] + ("..." if len(content) > 500 else ""))