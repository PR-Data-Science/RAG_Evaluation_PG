"""
PDF Ingestion Module for RAG System

This module handles PDF file discovery, text extraction, and chunking with metadata.
It processes all PDF files from a directory recursively and creates overlapping text chunks
suitable for retrieval-augmented generation systems.

No external API calls or embeddings are generated in this module.
"""

import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

try:
    import tiktoken
except ImportError:
    raise ImportError(
        "tiktoken is required for PDF ingestion. "
        "Install it with: pip install tiktoken"
    )

try:
    from pypdf import PdfReader
except ImportError:
    raise ImportError(
        "pypdf is required for PDF ingestion. "
        "Install it with: pip install pypdf"
    )

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)

# Constants
MODEL_NAME = "gpt-4o"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150


class PDFIngestor:
    """
    A production-grade PDF ingestion system for RAG pipelines.
    
    Handles PDF discovery, text extraction, cleaning, and chunking with metadata.
    Gracefully handles corrupted PDFs and maintains consistent output format.
    """
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        """
        Initialize the PDF ingestor.
        
        Args:
            model_name: Tokenizer model name (default: gpt-4o)
            chunk_size: Number of tokens per chunk (default: 500)
            chunk_overlap: Number of overlapping tokens between chunks (default: 150)
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize tiktoken encoder
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
            logger.info(f"[INIT] Tokenizer initialized for model: {model_name}")
        except KeyError:
            logger.warning(
                f"[INIT] Model {model_name} not found in tiktoken. "
                f"Using cl100k_base encoding instead."
            )
            self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # Validation
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"Chunk overlap ({self.chunk_overlap}) must be less than "
                f"chunk size ({self.chunk_size})"
            )
        
        self.chunks_created = 0
        logger.info(
            f"[INIT] PDFIngestor configured: "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )
    
    @staticmethod
    def _discover_pdfs(pdf_dir: str) -> List[Path]:
        """
        Recursively discover all PDF files in a directory.
        
        Args:
            pdf_dir: Path to directory containing PDFs
            
        Returns:
            List of Path objects for all discovered PDF files
            
        Raises:
            FileNotFoundError: If directory does not exist
        """
        pdf_path = Path(pdf_dir)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_dir}")
        
        if not pdf_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {pdf_dir}")
        
        # Recursively find all PDF files
        pdf_files = list(pdf_path.rglob("*.pdf"))
        
        logger.info(f"[DISCOVERY] Found {len(pdf_files)} PDF files in {pdf_dir}")
        
        if len(pdf_files) == 0:
            logger.warning(f"[DISCOVERY] No PDF files found in {pdf_dir}")
        
        return sorted(pdf_files)
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean extracted text by removing excessive whitespace and normalizing.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Normalize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Replace multiple newlines with double newline
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        # Remove excessive spaces on single lines
        lines = text.split('\n')
        lines = [' '.join(line.split()) for line in lines]
        text = '\n'.join(lines)
        
        # Strip leading and trailing whitespace
        text = text.strip()
        
        return text
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> List[Tuple[str, int]]:
        """
        Extract text from a PDF file page-by-page.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of tuples (page_text, page_number)
            
        Raises:
            Exception: If PDF reading fails (logged but not raised to caller)
        """
        pages_data = []
        
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            
            logger.info(f"[PDF] Reading {pdf_path.name}: {total_pages} pages")
            
            for page_num, page in enumerate(reader.pages):
                try:
                    # Extract text from page
                    text = page.extract_text()
                    
                    # Clean text
                    text = self._clean_text(text)
                    
                    # Skip empty pages
                    if text.strip():
                        pages_data.append((text, page_num + 1))
                        logger.debug(f"[PDF] Extracted page {page_num + 1}/{total_pages}")
                    else:
                        logger.debug(f"[PDF] Skipped empty page {page_num + 1}")
                
                except Exception as e:
                    logger.warning(
                        f"[PDF] Error extracting text from page {page_num + 1} "
                        f"in {pdf_path.name}: {str(e)}"
                    )
                    continue
            
            logger.info(
                f"[PDF] Successfully extracted {len(pages_data)} non-empty pages "
                f"from {pdf_path.name}"
            )
            
        except Exception as e:
            logger.error(
                f"[PDF] Failed to read PDF file {pdf_path.name}: {str(e)}. "
                f"Skipping this file."
            )
        
        return pages_data
    
    def _create_chunks(
        self,
        text: str,
        source: str,
        page: int
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with metadata.
        
        Args:
            text: Text to chunk
            source: Source PDF filename
            page: Page number
            
        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []
        
        # Tokenize the text
        tokens = self.encoder.encode(text)
        
        # Calculate stride (distance between chunk starts)
        stride = self.chunk_size - self.chunk_overlap
        
        # Create overlapping chunks
        for i in range(0, len(tokens), stride):
            # Get chunk tokens
            chunk_tokens = tokens[i : i + self.chunk_size]
            
            # Skip if chunk is too small (less than 10% of chunk size)
            if len(chunk_tokens) < self.chunk_size * 0.1:
                continue
            
            # Decode tokens back to text
            chunk_text = self.encoder.decode(chunk_tokens)
            
            # Create chunk dictionary with metadata
            chunk = {
                "id": str(uuid.uuid4()),
                "content": chunk_text,
                "metadata": {
                    "source": source,
                    "page": page,
                    "chunk_index": len(chunks),
                    "chunk_size": len(chunk_tokens),
                    "total_chunks": None  # Will be updated later
                }
            }
            
            chunks.append(chunk)
            self.chunks_created += 1
        
        # Update total_chunks for all chunks from this page
        total = len(chunks)
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = total
        
        logger.debug(
            f"[CHUNKING] Created {len(chunks)} chunks from "
            f"{source} page {page}"
        )
        
        return chunks
    
    def load_pdfs(self, pdf_dir: str) -> List[Dict[str, Any]]:
        """
        Load and process all PDFs from a directory.
        
        Discovers PDF files recursively, extracts text page-by-page, cleans,
        and creates overlapping chunks with metadata suitable for RAG systems.
        
        Args:
            pdf_dir: Path to directory containing PDF files
            
        Returns:
            List of chunk dictionaries, each containing:
                - id: Unique identifier (UUID)
                - content: Chunk text
                - metadata: Dict with source, page, chunk_index, etc.
                
        Raises:
            FileNotFoundError: If directory does not exist
            NotADirectoryError: If path is not a directory
        """
        logger.info(f"\n[START] Beginning PDF ingestion from: {pdf_dir}")
        
        # Reset counter
        self.chunks_created = 0
        all_chunks = []
        
        # Discover PDF files
        pdf_files = self._discover_pdfs(pdf_dir)
        
        if not pdf_files:
            logger.warning("[INGESTION] No PDF files found to process")
            return []
        
        # Process each PDF
        for pdf_path in pdf_files:
            try:
                logger.info(f"[PROCESSING] {pdf_path.name}")
                
                # Extract text page-by-page
                pages_data = self._extract_text_from_pdf(pdf_path)
                
                if not pages_data:
                    logger.warning(f"[PROCESSING] No text extracted from {pdf_path.name}")
                    continue
                
                # Create chunks from each page
                for page_text, page_num in pages_data:
                    chunks = self._create_chunks(
                        text=page_text,
                        source=pdf_path.name,
                        page=page_num
                    )
                    all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(
                    f"[PROCESSING] Unexpected error processing {pdf_path.name}: "
                    f"{str(e)}"
                )
                continue
        
        # Summary
        logger.info(
            f"\n[COMPLETE] Ingestion complete:\n"
            f"  - PDF files processed: {len(pdf_files)}\n"
            f"  - Total chunks created: {self.chunks_created}\n"
            f"  - Average chunk size: {self.chunk_size} tokens\n"
            f"  - Chunk overlap: {self.chunk_overlap} tokens"
        )
        
        return all_chunks


def load_pdfs(pdf_dir: str) -> List[Dict[str, Any]]:
    """
    Load and chunk PDF files from a directory (convenience function).
    
    This is the primary public function for PDF ingestion.
    
    Args:
        pdf_dir: Path to directory containing PDF files (searched recursively)
        
    Returns:
        List of chunk dictionaries with metadata, ready for RAG systems
        
    Example:
        >>> chunks = load_pdfs("./DataSources/UTA_HR_policies")
        >>> print(f"Loaded {len(chunks)} chunks")
        >>> print(chunks[0].keys())
        dict_keys(['id', 'content', 'metadata'])
    """
    ingestor = PDFIngestor(
        model_name=MODEL_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    return ingestor.load_pdfs(pdf_dir)


if __name__ == "__main__":
    # Example usage
    import json
    
    # Load PDFs from DataSources
    data_sources_path = "../../DataSources/UTA_HR_policies"
    
    if os.path.exists(data_sources_path):
        chunks = load_pdfs(data_sources_path)
        
        print(f"\n✓ Successfully loaded {len(chunks)} chunks")
        
        if chunks:
            print(f"\nFirst chunk structure:")
            print(json.dumps(chunks[0], indent=2, default=str))
    else:
        print(f"DataSources path not found: {data_sources_path}")
