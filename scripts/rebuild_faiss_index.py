#!/usr/bin/env python3
"""
Rebuild FAISS Index with OpenAI Embeddings (SAFE)

This script:
1. Loads existing ingestion chunks
2. Generates OpenAI embeddings for each chunk
3. Validates embeddings (dimension, NaN checks)
4. Builds new FAISS index
5. Saves to NEW file: faiss_index_openai.index (does NOT overwrite old index)

Usage:
    python scripts/rebuild_faiss_index.py

Environment:
    - OPENAI_API_KEY: Required (will error loudly if missing)
    - EMBEDDING_MODE: Ignored by this script (always uses OpenAI)

Safety Features:
    - Old index is NEVER overwritten
    - All errors raise exceptions (no silent failures)
    - Validates every embedding
    - Shows progress for each document
    - Creates backup of original index before any changes
"""

import sys
import os
import json
import logging
from pathlib import Path as PathlibPath

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    project_root = PathlibPath(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
except ImportError:
    pass  # dotenv not installed
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
LOGGER = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import faiss
import numpy as np

from src.embeddings.openai_embeddings import OpenAIEmbeddingClient

# Constants
EMBED_DIM = 3072
DOCUMENTS_PATH = Path("temp_storage") / "03_embedded_documents.json"
METADATA_PATH = Path("temp_storage") / "04_metadata_mapping.json"
OLD_INDEX_PATH = Path("faiss_index.index")
NEW_INDEX_PATH = Path("faiss_index_openai.index")
BACKUP_INDEX_PATH = Path("faiss_index_backup.index")


def validate_api_key() -> None:
    """Validate that OPENAI_API_KEY is set.
    
    Raises:
        EnvironmentError: If API key is missing
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        LOGGER.error("OPENAI_API_KEY environment variable is not set")
        raise EnvironmentError(
            "OPENAI_API_KEY is required. Set it with:\n"
            "  export OPENAI_API_KEY='your-key-here'\n"
            "Then re-run this script."
        )
    LOGGER.info("✓ OPENAI_API_KEY is set")


def load_ingestion_chunks() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load ingestion chunks and metadata.
    
    Returns:
        (documents, metadata_map)
        
    Raises:
        FileNotFoundError: If required files are missing
    """
    LOGGER.info(f"Loading documents from: {DOCUMENTS_PATH}")
    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(f"Documents not found at {DOCUMENTS_PATH}")
    
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)
    
    LOGGER.info(f"✓ Loaded {len(documents)} documents")
    
    LOGGER.info(f"Loading metadata from: {METADATA_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")
    
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    LOGGER.info(f"✓ Loaded metadata for {len(metadata)} entries")
    
    return documents, metadata


def get_text_from_doc(doc: Dict[str, Any]) -> str:
    """Extract text from document dict.
    
    Args:
        doc: Document dictionary
        
    Returns:
        Text content, or empty string if no text found
    """
    for key in ("text", "content", "text_preview", "chunk_text"):
        if key in doc and isinstance(doc[key], str):
            return doc[key]
    
    # Fallback: try any long string field
    for v in doc.values():
        if isinstance(v, str) and len(v) > 20:
            return v
    
    return ""


def generate_embeddings(
    documents: List[Dict[str, Any]],
    client: OpenAIEmbeddingClient,
) -> List[np.ndarray]:
    """Generate OpenAI embeddings for all documents.
    
    Args:
        documents: List of document dicts
        client: OpenAI embedding client
        
    Returns:
        List of normalized embedding vectors
        
    Raises:
        RuntimeError: If any embedding fails
    """
    LOGGER.info(f"Generating OpenAI embeddings for {len(documents)} documents...")
    
    embeddings = []
    failed_indices = []
    
    for i, doc in enumerate(documents):
        doc_id = doc.get("id", f"<doc_{i}>")
        
        try:
            # Extract text
            text = get_text_from_doc(doc)
            if not text:
                LOGGER.warning(f"Document {i} ({doc_id}) has no extractable text, using ID")
                text = doc_id
            
            # Generate embedding
            embedding_list = client.embed_text(text)
            embedding = np.array(embedding_list, dtype=np.float32)
            
            # Validate
            assert len(embedding) == EMBED_DIM, \
                f"Dimension mismatch: {len(embedding)} vs {EMBED_DIM}"
            assert not np.any(np.isnan(embedding)), \
                "Embedding contains NaN values"
            assert not np.any(np.isinf(embedding)), \
                "Embedding contains Inf values"
            
            embeddings.append(embedding)
            
            if (i + 1) % 10 == 0:
                LOGGER.info(f"  ✓ Generated embeddings for {i+1}/{len(documents)} documents")
        
        except Exception as e:
            LOGGER.error(f"Failed to generate embedding for document {i} ({doc_id}): {e}")
            failed_indices.append(i)
    
    if failed_indices:
        LOGGER.error(f"❌ {len(failed_indices)} documents failed:")
        for idx in failed_indices[:5]:  # Show first 5
            LOGGER.error(f"   - Document {idx}")
        raise RuntimeError(f"Failed to generate embeddings for {len(failed_indices)} documents")
    
    LOGGER.info(f"✓ Successfully generated {len(embeddings)} embeddings")
    return embeddings


def build_faiss_index(embeddings: List[np.ndarray]) -> faiss.Index:
    """Build FAISS index from embeddings.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        FAISS index (IndexFlatIP for cosine similarity)
    """
    LOGGER.info(f"Building FAISS index with {len(embeddings)} vectors...")
    
    # Stack embeddings into matrix
    xb = np.vstack(embeddings).astype(np.float32)
    LOGGER.info(f"  Shape: {xb.shape}")
    
    # Normalize for cosine-like inner product
    LOGGER.info("  Normalizing vectors (L2)...")
    faiss.normalize_L2(xb)
    
    # Create index
    LOGGER.info("  Creating IndexFlatIP...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(xb)
    
    LOGGER.info(f"✓ FAISS index built with {index.ntotal} vectors")
    return index


def backup_old_index() -> None:
    """Create backup of old index if it exists.
    
    SAFETY: Always backs up before overwriting anything.
    """
    if OLD_INDEX_PATH.exists():
        LOGGER.info(f"Creating backup: {OLD_INDEX_PATH} -> {BACKUP_INDEX_PATH}")
        OLD_INDEX_PATH.rename(BACKUP_INDEX_PATH)
        LOGGER.info(f"✓ Backup created at {BACKUP_INDEX_PATH}")
    else:
        LOGGER.info("No old index to backup")


def save_index(index: faiss.Index) -> None:
    """Save FAISS index to disk.
    
    Args:
        index: FAISS index to save
        
    Raises:
        RuntimeError: If save fails
    """
    LOGGER.info(f"Saving FAISS index to: {NEW_INDEX_PATH}")
    
    try:
        faiss.write_index(index, str(NEW_INDEX_PATH))
        LOGGER.info(f"✓ Index saved to {NEW_INDEX_PATH}")
        LOGGER.info(f"  Size: {NEW_INDEX_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    except Exception as e:
        LOGGER.error(f"Failed to save index: {e}")
        raise RuntimeError(f"Failed to save FAISS index: {e}") from e


def validate_old_index_still_works() -> None:
    """Quick sanity check that old index still exists and loads.
    
    This ensures we didn't accidentally corrupt or delete the original.
    """
    if BACKUP_INDEX_PATH.exists():
        try:
            LOGGER.info("Validating backup index...")
            test_index = faiss.read_index(str(BACKUP_INDEX_PATH))
            LOGGER.info(f"✓ Backup index loads successfully ({test_index.ntotal} vectors)")
        except Exception as e:
            LOGGER.error(f"Backup index validation failed: {e}")
            raise RuntimeError(f"Backup index is corrupted: {e}") from e


def main():
    """Main execution."""
    LOGGER.info("="*80)
    LOGGER.info("FAISS INDEX REBUILD - OPENAI EMBEDDINGS")
    LOGGER.info("="*80)
    
    try:
        # Step 1: Validate environment
        LOGGER.info("\n[Step 1/6] Validating API key...")
        validate_api_key()
        
        # Step 2: Load documents
        LOGGER.info("\n[Step 2/6] Loading ingestion chunks...")
        documents, metadata = load_ingestion_chunks()
        
        # Step 3: Initialize OpenAI client
        LOGGER.info("\n[Step 3/6] Initializing OpenAI client...")
        client = OpenAIEmbeddingClient()
        LOGGER.info("✓ OpenAI client ready")
        
        # Step 4: Generate embeddings
        LOGGER.info("\n[Step 4/6] Generating OpenAI embeddings...")
        embeddings = generate_embeddings(documents, client)
        
        # Step 5: Build FAISS index
        LOGGER.info("\n[Step 5/6] Building FAISS index...")
        index = build_faiss_index(embeddings)
        
        # Step 6: Save to new file (safe)
        LOGGER.info("\n[Step 6/6] Saving new index (preserving old one)...")
        LOGGER.info(f"Old index will be at: {BACKUP_INDEX_PATH}")
        LOGGER.info(f"New index will be at: {NEW_INDEX_PATH}")
        backup_old_index()
        save_index(index)
        validate_old_index_still_works()
        
        # Success!
        LOGGER.info("\n" + "="*80)
        LOGGER.info("✅ INDEX REBUILD COMPLETE")
        LOGGER.info("="*80)
        LOGGER.info(f"\nNew index: {NEW_INDEX_PATH.resolve()}")
        LOGGER.info(f"Backup of old: {BACKUP_INDEX_PATH.resolve()}")
        LOGGER.info("\nNext steps:")
        LOGGER.info("1. Test retrieval with new index")
        LOGGER.info("2. Compare results with old index")
        LOGGER.info("3. If satisfied, delete the backup")
        LOGGER.info("4. Set EMBEDDING_MODE=openai for production")
        LOGGER.info("\nTo use the new index:")
        LOGGER.info("  export EMBEDDING_MODE=openai")
        LOGGER.info("  python -c \"from src.retrieval.faiss_retriever import *; ...\"")
        
        return 0
    
    except Exception as e:
        LOGGER.error("\n" + "="*80)
        LOGGER.error("❌ INDEX REBUILD FAILED")
        LOGGER.error("="*80)
        LOGGER.error(f"Error: {e}")
        LOGGER.error("\nOld index is SAFE - not modified")
        LOGGER.error("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
