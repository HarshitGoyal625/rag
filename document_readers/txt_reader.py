"""Text file reader implementation."""
from pathlib import Path
from .base import DocumentReader


class TxtReader(DocumentReader):
    """Reader for plain text files."""

    def read(self, file_path: Path) -> str:
        """Read a text file and return its content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 for files with different encodings
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to read text file {file_path}: {e}")

    def supported_extensions(self) -> list[str]:
        """Return supported text file extensions."""
        return ['.txt', '.md', '.markdown', '.rst', '.csv', '.json', '.xml', '.html', '.htm']