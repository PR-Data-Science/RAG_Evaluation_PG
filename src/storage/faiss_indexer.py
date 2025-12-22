"""FAISS vector index builder and loader for document embeddings.

This module only handles vector indexing and storage. It does NOT:
- Generate embeddings
- Call OpenAI APIs
- Perform similarity search
- Manage metadata beyond preservation

Public API
----------
def build_faiss_index(docs: List[Dict], index_path: str) -> None
    Builds a FAISS index from documents with embeddings and saves to disk.

def load_faiss_index(index_path: str) -> tuple
    Loads a FAISS index and metadata from disk.

Requirements enforced by implementation:
- Uses FAISS with IndexFlatIP (cosine similarity)
- Normalizes vectors before indexing (L2 normalization)
- Stores index and metadata separately on disk
- Preserves doc_id, source, page, chunk_index, text
"""

from __future__ import annotations

import os
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

try:
    import faiss
    import numpy as np
except ImportError as exc:
    raise ImportError("faiss and numpy are required. Install with `pip install faiss-cpu numpy`") from exc

LOGGER = logging.getLogger(__name__)

# File names for index storage
INDEX_FILENAME = "faiss.index"
METADATA_FILENAME = "metadata.json"
CONFIG_FILENAME = "config.json"


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length (L2 normalization).

    Parameters
    ----------
    vectors:
        Array of shape (n_samples, n_features).

    Returns
    -------
    Normalized vectors of same shape.
    """
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norm[norm == 0] = 1.0
    return vectors / norm


def _validate_documents(docs: List[Dict[str, Any]]) -> None:
    """Validate that all documents have required keys and embeddings.

    Parameters
    ----------
    docs:
        List of document dictionaries.

    Raises
    ------
    ValueError:
        If any document is missing required keys or embedding.
    """
    for idx, doc in enumerate(docs):
        if not isinstance(doc, dict):
            raise ValueError(f"Document at index {idx} is not a dict")
        if "id" not in doc:
            raise ValueError(f"Document at index {idx} missing required 'id' key")
        if "embedding" not in doc:
            raise ValueError(f"Document at index {idx} missing required 'embedding' key")
        if "text" not in doc:
            raise ValueError(f"Document at index {idx} missing required 'text' key")
        if "metadata" not in doc:
            raise ValueError(f"Document at index {idx} missing required 'metadata' key")

        emb = doc["embedding"]
        if not isinstance(emb, (list, np.ndarray)):
            raise ValueError(f"Document at index {idx} embedding must be list or ndarray")
        if len(emb) == 0:
            raise ValueError(f"Document at index {idx} embedding is empty")


def build_faiss_index(docs: List[Dict[str, Any]], index_path: str) -> None:
    """Builds and saves a FAISS index from documents with embeddings.

    Parameters
    ----------
    docs:
        List of dictionaries with keys:
        - 'id': unique document identifier (str)
        - 'text': original text content (str)
        - 'embedding': vector as list or ndarray of floats
        - 'metadata': dict with source, page, chunk_index, etc.

    index_path:
        Directory path to save FAISS index and metadata.

    Notes
    -----
    - Vectors are L2 normalized before indexing (required for cosine similarity).
    - Index uses IndexFlatIP (inner product, which is cosine similarity on normalized vectors).
    - Creates subdirectory if it does not exist.
    - Saves metadata separately as JSON for document tracking.
    - Saves config.json with vector dimension and document count.

    Raises
    ------
    ValueError:
        If documents are invalid or missing required keys.
    """
    if not isinstance(docs, list) or len(docs) == 0:
        raise ValueError("docs must be a non-empty list")

    _validate_documents(docs)

    index_dir = Path(index_path)
    index_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Building FAISS index for %d documents into %s", len(docs), index_path)

    # Extract embeddings and normalize
    embeddings = np.array([doc["embedding"] for doc in docs], dtype=np.float32)
    LOGGER.debug("Extracted embeddings shape: %s", embeddings.shape)

    embeddings = _normalize_vectors(embeddings)
    LOGGER.debug("Normalized embeddings for cosine similarity")

    # Create FAISS index with inner product (cosine similarity on normalized vectors)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    LOGGER.info("Created IndexFlatIP with dimension %d and %d vectors", dimension, len(docs))

    # Save index to disk
    index_file = index_dir / INDEX_FILENAME
    faiss.write_index(index, str(index_file))
    LOGGER.info("Saved FAISS index to %s", index_file)

    # Build metadata mapping (id -> metadata + text)
    metadata_map = {}
    for doc in docs:
        metadata_map[doc["id"]] = {
            "text": doc["text"],
            "source": doc["metadata"].get("source"),
            "page": doc["metadata"].get("page"),
            "chunk_index": doc["metadata"].get("chunk_index"),
            "chunk_size": doc["metadata"].get("chunk_size"),
            "total_chunks": doc["metadata"].get("total_chunks"),
            "custom_metadata": {k: v for k, v in doc["metadata"].items() if k not in ["source", "page", "chunk_index", "chunk_size", "total_chunks"]},
        }

    # Save metadata to JSON
    metadata_file = index_dir / METADATA_FILENAME
    with open(metadata_file, "w") as f:
        json.dump(metadata_map, f, indent=2)
    LOGGER.info("Saved metadata for %d documents to %s", len(metadata_map), metadata_file)

    # Save config
    config = {
        "dimension": dimension,
        "n_documents": len(docs),
        "index_type": "IndexFlatIP",
        "normalized": True,
    }
    config_file = index_dir / CONFIG_FILENAME
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    LOGGER.info("Saved config to %s", config_file)

    LOGGER.info("FAISS index build complete: %d documents, dimension %d", len(docs), dimension)


def load_faiss_index(index_path: str) -> Tuple[Any, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Loads a FAISS index and metadata from disk.

    Parameters
    ----------
    index_path:
        Directory path where FAISS index and metadata were saved.

    Returns
    -------
    Tuple of:
    - index: FAISS index object
    - metadata_map: dict mapping document ID -> metadata
    - config: dict with index configuration (dimension, n_documents, etc.)

    Raises
    ------
    FileNotFoundError:
        If index files do not exist.
    ValueError:
        If index or metadata files are corrupted.
    """
    index_dir = Path(index_path)
    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory not found: {index_path}")

    index_file = index_dir / INDEX_FILENAME
    metadata_file = index_dir / METADATA_FILENAME
    config_file = index_dir / CONFIG_FILENAME

    if not index_file.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found at {config_file}")

    LOGGER.info("Loading FAISS index from %s", index_path)

    # Load FAISS index
    try:
        index = faiss.read_index(str(index_file))
        LOGGER.info("Loaded FAISS index: %d vectors", index.ntotal)
    except Exception as exc:
        LOGGER.exception("Failed to load FAISS index")
        raise ValueError(f"Corrupted FAISS index at {index_file}") from exc

    # Load metadata
    try:
        with open(metadata_file, "r") as f:
            metadata_map = json.load(f)
        LOGGER.info("Loaded metadata for %d documents", len(metadata_map))
    except Exception as exc:
        LOGGER.exception("Failed to load metadata")
        raise ValueError(f"Corrupted metadata file at {metadata_file}") from exc

    # Load config
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        LOGGER.info("Loaded config: dimension=%d, n_documents=%d", config.get("dimension"), config.get("n_documents"))
    except Exception as exc:
        LOGGER.exception("Failed to load config")
        raise ValueError(f"Corrupted config file at {config_file}") from exc

    # Validate consistency
    if index.ntotal != len(metadata_map):
        LOGGER.warning(
            "Vector count mismatch: index has %d vectors but metadata has %d entries",
            index.ntotal,
            len(metadata_map),
        )

    LOGGER.info("Successfully loaded FAISS index from %s", index_path)
    return index, metadata_map, config


__all__ = ["build_faiss_index", "load_faiss_index"]
