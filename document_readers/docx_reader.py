"""DOCX file reader implementation."""
from pathlib import Path
from .base import DocumentReader


class DocxReader(DocumentReader):
    """Reader for Microsoft Word DOCX files using python-docx."""

    def read(self, file_path: Path) -> str:
        """Read a DOCX file and return its text content."""
        try:
            from docx import Document
        except ImportError:
            raise ValueError(
                "python-docx is required to read DOCX files. Install with: pip install python-docx"
            )

        text_parts = []
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_parts.append(' | '.join(row_text))
        except (FileNotFoundError, PermissionError):
            raise  # Let file-system errors propagate as-is
        except Exception as e:
            raise ValueError(f"Failed to read DOCX file {file_path}: {e}")
        return '\n\n'.join(text_parts)

    def supported_extensions(self) -> list[str]:
        """Return supported DOCX file extensions."""
        # Note: .doc (legacy Word) is NOT supported by python-docx
        # Use antiword, textract, or convert to .docx first
        return ['.docx']