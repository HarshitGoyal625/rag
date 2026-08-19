# RAG Architecture


# Document Reading (part 1)
A modular document processing pipeline that extracts text from various file formats using the **Factory Pattern** and **Strategy Pattern**.

## Architecture Overview

```
main.py → DocumentReaderFactory → Concrete Readers (TxtReader, PdfReader, DocxReader)
                ↑
         Abstract Base (DocumentReader)
```

### Components

| File | Role |
|------|------|
| `main.py` | Entry point - recursively scans directory, delegates to factory |
| `document_readers/base.py` | Abstract `DocumentReader` class defining the interface |
| `document_readers/factory.py` | `DocumentReaderFactory` - creates/manages readers, routes files |
| `document_readers/txt_reader.py` | Reads `.txt`, `.md`, `.json`, `.csv`, `.xml`, `.html`, etc. |
| `document_readers/pdf_reader.py` | Reads `.pdf` using `pypdf` |
| `document_readers/docx_reader.py` | Reads `.docx` using `python-docx` (tables included) |

### Key Design Patterns

- **Factory Pattern**: `DocumentReaderFactory` instantiates and selects the right reader based on file extension
- **Strategy Pattern**: Each reader implements the same `DocumentReader` interface (`read()`, `supported_extensions()`)

# Chunking the docs (part 2)
