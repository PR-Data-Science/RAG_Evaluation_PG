"""
UI module for RAG system evaluation and testing.

This package contains the production-grade Gradio interface for testing
and evaluating the Retrieval-Augmented Generation (RAG) system.
"""

from src.ui.rag_evaluation_ui import (
    run_rag_pipeline,
    build_context,
    generate_rag_answer,
    evaluate_response,
    log_failure,
)

__all__ = [
    "run_rag_pipeline",
    "build_context",
    "generate_rag_answer",
    "evaluate_response",
    "log_failure",
]
