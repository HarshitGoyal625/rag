"""Factory for creating document readers based on file extension."""
from pathlib import Path
from typing import Optional
from .base import DocumentReader
from .txt_reader import TxtReader
from .pdf_reader import PdfReader
from .docx_reader import DocxReader


class DocumentReaderFactory:
    """Factory for creating appropriate document readers."""

    _readers: list[DocumentReader] = []

    @classmethod
    def _get_readers(cls) -> list[DocumentReader]:
        """Get or initialize the list of readers (lazy initialization)."""
        if not cls._readers:
            cls._readers = [
                TxtReader(),
                PdfReader(),
                DocxReader(),
            ]
        return cls._readers

    @classmethod
    def get_reader(cls, file_path: Path) -> Optional[DocumentReader]:
        """
        Get the appropriate reader for a file.

        Args:
            file_path: Path to the file

        Returns:
            DocumentReader instance if supported, None otherwise
        """
        for reader in cls._get_readers():
            if reader.can_read(file_path):
                return reader
        return None

    @classmethod
    def read_document(cls, file_path: Path) -> str:
        """
        Read a document using the appropriate reader.

        Args:
            file_path: Path to the document file

        Returns:
            Extracted text content

        Raises:
            ValueError: If no reader supports the file type or reading fails
        """
        reader = cls.get_reader(file_path)
        if reader is None:
            supported = cls.get_supported_extensions()
            raise ValueError(
                f"No reader available for {file_path.suffix}. "
                f"Supported extensions: {supported}"
            )
        return reader.read(file_path)

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Get all supported file extensions."""
        extensions = []
        for reader in cls._get_readers():
            extensions.extend(reader.supported_extensions())
        return sorted(set(extensions))