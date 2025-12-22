"""
Ingestion package for RAG system.

This package handles document ingestion, text extraction, and chunking.
"""

from .pdf_ingestor import load_pdfs, PDFIngestor

__all__ = ["load_pdfs", "PDFIngestor"]
