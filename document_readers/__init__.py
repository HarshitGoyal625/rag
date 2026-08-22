"""Document reader factory and implementations."""
from .base import DocumentReader
from .structure import DocumentStructure, Block, BlockType
from .txt_reader import TxtReader
from .pdf_reader import PdfReader
from .docx_reader import DocxReader
from .factory import DocumentReaderFactory

__all__ = [
    "DocumentReader",
    "DocumentStructure",
    "Block",
    "BlockType",
    "TxtReader",
    "PdfReader",
    "DocxReader",
    "DocumentReaderFactory",
]