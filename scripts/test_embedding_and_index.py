#!/usr/bin/env python
"""Test script for embedding generation and FAISS indexing.

This script:
1. Loads chunks from STEP 1 (PDF ingestion)
2. Generates embeddings from STEP 2A (embedding generator)
3. Builds FAISS index from STEP 2B (FAISS indexer)

Run with:
    export OPENAI_API_KEY="sk-..."
    ./.venv/bin/python scripts/test_embedding_and_index.py
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import load_pdfs
from src.embeddings import embed_documents
from src.storage import build_faiss_index, load_faiss_index


def main():
    """Run the full pipeline test."""
    LOGGER.info("=" * 60)
    LOGGER.info("Testing STEP 1→2A→2B: Ingestion → Embeddings → FAISS")
    LOGGER.info("=" * 60)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        LOGGER.error("OPENAI_API_KEY environment variable is not set")
        LOGGER.info("To test embeddings, run: export OPENAI_API_KEY='sk-...'")
        LOGGER.info("Continuing with mock test instead...")
        test_with_mock_embeddings()
        return

    # STEP 1: Load PDFs
    LOGGER.info("\n[STEP 1] Loading PDF chunks...")
    pdf_dir = "DataSources/UTA_HR_policies"
    if not os.path.exists(pdf_dir):
        LOGGER.error("PDF directory not found: %s", pdf_dir)
        LOGGER.info("Please run this script from the project root directory.")
        return

    chunks = load_pdfs(pdf_dir)
    LOGGER.info("✓ Loaded %d chunks", len(chunks))

    # Sample a few chunks for embedding (to save API calls during testing)
    sample_chunks = chunks[:5]  # Use only first 5 chunks for testing
    LOGGER.info("✓ Using %d chunks for testing", len(sample_chunks))

    # Convert chunks to embedding format
    docs_for_embedding = []
    for chunk in sample_chunks:
        docs_for_embedding.append({
            "id": chunk["id"],
            "text": chunk["content"],
            "metadata": chunk["metadata"],
        })

    # STEP 2A: Generate embeddings
    LOGGER.info("\n[STEP 2A] Generating embeddings...")
    try:
        embedded_docs = embed_documents(docs_for_embedding)
        LOGGER.info("✓ Generated embeddings for %d documents", len(embedded_docs))
        LOGGER.info("✓ Embedding dimension: %d", len(embedded_docs[0]["embedding"]))
    except Exception as e:
        LOGGER.error("Failed to generate embeddings: %s", e)
        LOGGER.info("Make sure OPENAI_API_KEY is set correctly")
        return

    # STEP 2B: Build FAISS index
    LOGGER.info("\n[STEP 2B] Building FAISS index...")
    index_path = "data/faiss_index"
    try:
        build_faiss_index(embedded_docs, index_path)
        LOGGER.info("✓ Built FAISS index at %s", index_path)
    except Exception as e:
        LOGGER.error("Failed to build FAISS index: %s", e)
        return

    # Load and verify index
    LOGGER.info("\n[VERIFY] Loading index to verify...")
    try:
        index, metadata_map, config = load_faiss_index(index_path)
        LOGGER.info("✓ Loaded FAISS index: %d vectors", index.ntotal)
        LOGGER.info("✓ Loaded metadata: %d documents", len(metadata_map))
        LOGGER.info("✓ Index config: dimension=%d, type=%s", config["dimension"], config["index_type"])
    except Exception as e:
        LOGGER.error("Failed to load FAISS index: %s", e)
        return

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("✓ SUCCESS: Full pipeline test completed!")
    LOGGER.info("=" * 60)
    LOGGER.info("Index location: %s", index_path)
    LOGGER.info("Documents indexed: %d", index.ntotal)
    LOGGER.info("Vector dimension: %d", config["dimension"])


def test_with_mock_embeddings():
    """Test with mock embeddings (no API calls)."""
    LOGGER.info("\n[MOCK TEST] Running with synthetic embeddings...")
    import numpy as np

    # Create mock documents with embeddings
    mock_docs = [
        {
            "id": "mock-1",
            "text": "Sample document 1 about HR policies",
            "embedding": np.random.randn(3072).tolist(),  # text-embedding-3-large dimension
            "metadata": {"source": "mock.pdf", "page": 1, "chunk_index": 0},
        },
        {
            "id": "mock-2",
            "text": "Sample document 2 about benefits",
            "embedding": np.random.randn(3072).tolist(),
            "metadata": {"source": "mock.pdf", "page": 2, "chunk_index": 1},
        },
        {
            "id": "mock-3",
            "text": "Sample document 3 about leave policies",
            "embedding": np.random.randn(3072).tolist(),
            "metadata": {"source": "mock.pdf", "page": 3, "chunk_index": 2},
        },
    ]

    index_path = "data/faiss_index_mock"
    try:
        LOGGER.info("Building mock FAISS index...")
        build_faiss_index(mock_docs, index_path)
        LOGGER.info("✓ Built mock FAISS index")

        LOGGER.info("Loading mock FAISS index...")
        index, metadata_map, config = load_faiss_index(index_path)
        LOGGER.info("✓ Loaded mock index: %d vectors", index.ntotal)
        LOGGER.info("✓ Index config: dimension=%d", config["dimension"])
    except Exception as e:
        LOGGER.error("Mock test failed: %s", e)
        return

    LOGGER.info("\n✓ Mock test passed!")


if __name__ == "__main__":
    main()
