"""PDF file reader implementation."""
from pathlib import Path
from .base import DocumentReader


class PdfReader(DocumentReader):
    """Reader for PDF files using pypdf."""

    def read(self, file_path: Path) -> str:
        """Read a PDF file and return its text content."""
        try:
            import pypdf
        except ImportError:
            raise ValueError(
                "pypdf is required to read PDF files. Install with: pip install pypdf"
            )

        text_parts = []
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
        except (FileNotFoundError, PermissionError):
            raise  # Let file-system errors propagate as-is
        except Exception as e:
            raise ValueError(f"Failed to read PDF file {file_path}: {e}")
        return '\n\n'.join(text_parts)

    def supported_extensions(self) -> list[str]:
        """Return supported PDF file extensions."""
        return ['.pdf']