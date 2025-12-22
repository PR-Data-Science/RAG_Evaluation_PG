"""Storage module for RAG vector indexes and metadata."""

from .faiss_indexer import build_faiss_index, load_faiss_index

__all__ = ["build_faiss_index", "load_faiss_index"]
