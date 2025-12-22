"""
FAISS-based retrieval module for RAG system.

This package provides vector similarity search using FAISS to retrieve
relevant HR policy chunks based on semantic queries.
"""

from src.retrieval.faiss_retriever import (
    load_retrieval_assets,
    retrieve_top_k,
    audit_retrieval,
    embed_query,
    embed_from_document,
)

__all__ = [
    "load_retrieval_assets",
    "retrieve_top_k",
    "audit_retrieval",
    "embed_query",
    "embed_from_document",
]
