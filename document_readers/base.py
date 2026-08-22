"""Abstract base class for document readers."""
from abc import ABC, abstractmethod
from pathlib import Path

from .structure import DocumentStructure


class DocumentReader(ABC):
    """Abstract base class for reading documents into a structured IR.

    Subclasses convert a source file into a `DocumentStructure` instance
    that the (doc-type-agnostic) chunker consumes. `read()` returns the
    structured form; a thin `read_text()` helper is provided for code that
    still wants a flat-string view of the document.
    """

    @abstractmethod
    def read(self, file_path: Path) -> DocumentStructure:
        """
        Read a document and return it as a structured IR.

        Args:
            file_path: Path to the document file

        Returns:
            `DocumentStructure` preserving the source's structure
            (headings/levels, pages, tables, tables) on a best-effort
            basis for the format.

        Raises:
            ValueError: If the file cannot be read or is corrupted
        """
        pass

    def read_text(self, file_path: Path) -> str:
        """Convenience: return the document as flat text.

        Joins all block texts with double newlines. Useful for preserving
        backward compatibility or quick previews, but loses structure.
        """
        doc = self.read(file_path)
        return "\n\n".join(
            block.text for block in doc.blocks if block.text.strip()
        )

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