# PDF Ingestion Module Documentation

## Overview

The PDF Ingestion module (`src/ingestion/pdf_ingestor.py`) is STEP 1 of the RAG system. It handles:

- **PDF Discovery**: Recursively finds all PDF files in a directory (no hardcoding)
- **Text Extraction**: Extracts text page-by-page using pypdf
- **Text Cleaning**: Removes whitespace, normalizes newlines, skips empty pages
- **Text Chunking**: Creates overlapping chunks using tiktoken tokenization
- **Metadata Enrichment**: Adds source, page, and chunk information
- **Error Handling**: Gracefully handles corrupted PDFs

## Installation

Install required dependencies:

```bash
pip install pypdf tiktoken
```

## Usage

### Basic Usage (Recommended)

```python
from src.ingestion import load_pdfs

# Load all PDFs from a directory (recursive)
chunks = load_pdfs("DataSources/UTA_HR_policies")

# chunks is a List[Dict] with this structure:
# [
#   {
#     "id": "uuid-string",
#     "content": "chunk text...",
#     "metadata": {
#       "source": "filename.pdf",
#       "page": 1,
#       "chunk_index": 0,
#       "chunk_size": 487,
#       "total_chunks": 15
#     }
#   },
#   ...
# ]
```

### Advanced Usage (Custom Configuration)

```python
from src.ingestion import PDFIngestor

# Create custom ingestor
ingestor = PDFIngestor(
    model_name="gpt-4o",      # Tokenizer model
    chunk_size=500,            # Tokens per chunk
    chunk_overlap=150          # Overlapping tokens
)

# Load PDFs
chunks = ingestor.load_pdfs("DataSources/UTA_HR_policies")
```

## Directory Structure

The module automatically discovers PDFs in any subfolder structure:

```
DataSources/
├── UTA_HR_policies/
│   ├── Benefits/
│   │   ├── health_insurance.pdf
│   │   └── retirement.pdf
│   ├── Employment/
│   │   └── hiring_practices.pdf
│   ├── Leave & Absences/
│   └── Staff Performance & Evaluation/
```

**No hardcoding needed!** All PDFs are discovered recursively.

## Configuration

### Default Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | gpt-4o | Tokenizer model (from tiktoken) |
| Chunk Size | 500 | Tokens per chunk |
| Chunk Overlap | 150 | Overlapping tokens between chunks |
| Min Chunk Size | 50 | 10% of chunk_size (filters tiny chunks) |

### Customization Example

```python
from src.ingestion import PDFIngestor

# For smaller chunks
ingestor_small = PDFIngestor(chunk_size=300, chunk_overlap=75)

# For larger chunks
ingestor_large = PDFIngestor(chunk_size=1000, chunk_overlap=250)
```

## Output Format

Each chunk is a dictionary with:

```python
{
    "id": "550e8400-e29b-41d4-a716-446655440000",  # UUID
    "content": "Text of the chunk...",  # The actual chunk text
    "metadata": {
        "source": "health_insurance.pdf",  # PDF filename
        "page": 1,                          # Page number
        "chunk_index": 0,                   # Chunk sequence number
        "chunk_size": 487,                  # Actual tokens in this chunk
        "total_chunks": 15                  # Total chunks from this page
    }
}
```

## Features

### ✅ What It Does

- Reads PDFs recursively from any directory structure
- Extracts text page-by-page with error handling
- Cleans text (whitespace, newlines normalization)
- Skips empty pages automatically
- Creates overlapping token-based chunks
- Assigns unique IDs (UUID) to each chunk
- Includes rich metadata for tracking
- Logs all operations for debugging
- Handles corrupted PDFs gracefully

### ❌ What It Does NOT Do

- Generate embeddings (that's STEP 2)
- Use FAISS or vector stores (that's STEP 2)
- Call OpenAI APIs
- Build chatbots (that's later)
- Store data to disk

## Error Handling

The module gracefully handles various error scenarios:

```python
# ✓ Handles corrupted PDFs - logs and skips
# ✓ Handles missing directories - raises FileNotFoundError
# ✓ Handles page extraction errors - logs and continues
# ✓ Handles empty PDFs - logs warning and skips
# ✓ Handles invalid paths - raises NotADirectoryError
```

### Example: Handling Errors

```python
from src.ingestion import load_pdfs

try:
    chunks = load_pdfs("DataSources/UTA_HR_policies")
except FileNotFoundError as e:
    print(f"Directory not found: {e}")
except NotADirectoryError as e:
    print(f"Path is not a directory: {e}")

# Logs will show any per-file errors
```

## Logging

The module uses Python's logging module with the following levels:

- `[INFO]`: Main operations (discovery, processing, completion)
- `[WARNING]`: Recoverable issues (empty pages, file skips)
- `[ERROR]`: Serious issues (PDF read failures)
- `[DEBUG]`: Detailed information (page extraction, chunking)

Enable debug logging to see page-level details:

```python
import logging
logging.getLogger("pdf_ingestor").setLevel(logging.DEBUG)
```

## Performance Notes

- **Scalability**: Tested with 100+ PDFs
- **Memory**: Processes one PDF at a time
- **Speed**: Depends on PDF size and page count
- **Tokenization**: Uses cached tiktoken encoding

## Example Output

```python
chunks = load_pdfs("DataSources/UTA_HR_policies")

# Information logged:
# [INFO] pdf_ingestor: [START] Beginning PDF ingestion from: DataSources/UTA_HR_policies
# [INFO] pdf_ingestor: [DISCOVERY] Found 12 PDF files in DataSources/UTA_HR_policies
# [INFO] pdf_ingestor: [PROCESSING] health_insurance.pdf
# [INFO] pdf_ingestor: [PDF] Reading health_insurance.pdf: 45 pages
# [INFO] pdf_ingestor: [PDF] Successfully extracted 42 non-empty pages from health_insurance.pdf
# [INFO] pdf_ingestor: [COMPLETE] Ingestion complete:
#   - PDF files processed: 12
#   - Total chunks created: 548
#   - Average chunk size: 500 tokens
#   - Chunk overlap: 150 tokens

print(f"Total chunks: {len(chunks)}")  # 548
print(f"Sample chunk ID: {chunks[0]['id']}")
print(f"Sample chunk source: {chunks[0]['metadata']['source']}")
```

## Next Steps

After ingestion:
1. **STEP 2**: Generate embeddings for chunks (embedding model)
2. **STEP 3**: Store embeddings in vector database (FAISS/Pinecone)
3. **STEP 4**: Build retrieval system (semantic search)
4. **STEP 5**: Integrate with LLM (RAG pipeline)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'pypdf'` | Run `pip install pypdf` |
| `ModuleNotFoundError: No module named 'tiktoken'` | Run `pip install tiktoken` |
| `FileNotFoundError` | Check directory path exists |
| No PDFs found | Check PDF files are in directory or subdirectories |
| Memory issues | Process fewer PDFs at once (split directory) |
| Corrupted PDF warning | PDF may be scanned image - consider OCR |

## API Reference

### `load_pdfs(pdf_dir: str) -> List[Dict[str, Any]]`

Main entry point for PDF ingestion.

**Parameters:**
- `pdf_dir` (str): Path to directory containing PDFs

**Returns:**
- `List[Dict[str, Any]]`: List of chunk dictionaries

**Raises:**
- `FileNotFoundError`: If directory doesn't exist
- `NotADirectoryError`: If path is not a directory

### `PDFIngestor` Class

Advanced usage with custom configuration.

**Constructor:**
```python
PDFIngestor(
    model_name: str = "gpt-4o",
    chunk_size: int = 500,
    chunk_overlap: int = 150
)
```

**Methods:**
- `load_pdfs(pdf_dir: str) -> List[Dict[str, Any]]`

## Code Quality

- ✅ Full type hints
- ✅ Comprehensive docstrings
- ✅ Error handling
- ✅ Logging throughout
- ✅ Modular design
- ✅ No external API calls
- ✅ Production-ready
