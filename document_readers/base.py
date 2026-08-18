"""Abstract base class for document readers."""
from abc import ABC, abstractmethod
from pathlib import Path


class DocumentReader(ABC):
    """Abstract base class for reading documents and extracting text."""

    @abstractmethod
    def read(self, file_path: Path) -> str:
        """
        Read a document and return its text content.

        Args:
            file_path: Path to the document file

        Returns:
            Extracted text content as string

        Raises:
            ValueError: If the file cannot be read or is corrupted
        """
        pass

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """
        Return list of file extensions this reader supports.

        Returns:
            List of supported file extensions (e.g., ['.txt', '.md'])
        """
        pass

    def can_read(self, file_path: Path) -> bool:
        """
        Check if this reader can handle the given file.

        Args:
            file_path: Path to check

        Returns:
            True if the file extension is supported
        """
        return file_path.suffix.lower() in self.supported_extensions()