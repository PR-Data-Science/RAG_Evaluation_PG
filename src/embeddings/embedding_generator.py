"""Embedding generator for pre-chunked documents.

This module only generates embeddings for a list of already-prepared
documents. It does NOT read PDFs, chunk text, store vectors, or call any
vector database.

Public API
----------
def embed_documents(docs: List[Dict]) -> List[Dict]
    Adds an "embedding" key to each dict in `docs` using OpenAI embeddings.

Requirements enforced by implementation:
- Uses environment variable `OPENAI_API_KEY`
- Uses model `text-embedding-3-large`
- Batches inputs (configurable `batch_size`)
- Retries with exponential backoff on failures
- Logs progress and errors

"""
from __future__ import annotations

import os
import time
import random
import logging
from typing import List, Dict, Any, Optional

try:
    import openai
except Exception as exc:  # pragma: no cover - environment dependent
    raise ImportError("openai package is required. Install with `pip install openai`") from exc

LOGGER = logging.getLogger(__name__)


def _ensure_api_key() -> None:
    """Ensure `OPENAI_API_KEY` is available and configure `openai.api_key`.

    Raises:
        EnvironmentError: when the env var is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        LOGGER.error("Environment variable OPENAI_API_KEY is not set")
        raise EnvironmentError("OPENAI_API_KEY environment variable is required")
    openai.api_key = api_key


def _call_embeddings_batch(
    inputs: List[str],
    model: str = "text-embedding-3-large",
    max_retries: int = 5,
    initial_backoff: float = 1.0,
    timeout: Optional[float] = None,
) -> List[List[float]]:
    """Call OpenAI Embeddings API for a single batch with retries.

    Parameters
    - inputs: list of strings to embed
    - model: embedding model name
    - max_retries: number of attempts before failing
    - initial_backoff: base seconds for exponential backoff
    - timeout: optional request timeout passed to OpenAI client

    Returns list of embedding vectors (list of floats) in same order.
    """
    attempt = 0
    last_exception = None
    while attempt < max_retries:
        try:
            attempt += 1
            LOGGER.debug("Embedding batch call attempt %d, size=%d", attempt, len(inputs))
            # openai.Embedding.create accepts `input` as list of strings
            kwargs: Dict[str, Any] = {"model": model, "input": inputs}
            if timeout is not None:
                kwargs["request_timeout"] = timeout
            resp = openai.Embedding.create(**kwargs)
            data = resp.get("data") or []
            embeddings = [d["embedding"] for d in data]
            if len(embeddings) != len(inputs):
                LOGGER.warning(
                    "Embeddings returned length mismatch: got %d embeddings for %d inputs",
                    len(embeddings),
                    len(inputs),
                )
            return embeddings
        except Exception as exc:  # broad catch to handle rate limits, timeouts, etc.
            last_exception = exc
            sleep_for = initial_backoff * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            LOGGER.warning(
                "Embedding API call failed on attempt %d/%d: %s. Retrying in %.2fs",
                attempt,
                max_retries,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)

    LOGGER.exception("All embedding attempts failed after %d tries", max_retries)
    raise RuntimeError("Failed to obtain embeddings") from last_exception


def embed_documents(
    docs: List[Dict[str, Any]],
    model: str = "text-embedding-3-large",
    batch_size: int = 128,
    max_retries: int = 5,
    initial_backoff: float = 1.0,
    timeout: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Adds embeddings to pre-chunked documents.

    Parameters
    ----------
    docs:
        List of dictionaries with keys: `id` (str), `text` (str), `metadata` (dict).
        Each dict will get an extra key `embedding` with a list of floats.
    model:
        OpenAI embedding model to use. Default: `text-embedding-3-large`.
    batch_size:
        Number of documents to send per API call. Default: 128.
    max_retries:
        Number of retry attempts for transient API failures.
    initial_backoff:
        Initial backoff seconds for exponential retry.
    timeout:
        Optional timeout (seconds) passed to OpenAI client.

    Returns
    -------
    The same `docs` list (modified in-place) with `embedding` added to each dict.

    Notes
    -----
    - This function only generates embeddings and does not perform any
      I/O other than calling the OpenAI API.
    - The function will raise `EnvironmentError` if `OPENAI_API_KEY` is missing.
    """
    if not isinstance(docs, list):
        raise ValueError("`docs` must be a list of dictionaries")

    _ensure_api_key()

    total = len(docs)
    LOGGER.info("Embedding %d documents with model=%s (batch_size=%d)", total, model, batch_size)

    # Validate and collect texts in order
    for idx, d in enumerate(docs):
        if not isinstance(d, dict):
            raise ValueError(f"Document at index {idx} is not a dict")
        if "text" not in d:
            raise ValueError(f"Document at index {idx} missing required 'text' key")

    # Process in batches
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = docs[start:end]
        texts = [doc["text"] for doc in batch]
        LOGGER.info("Requesting embeddings for documents %d-%d", start, end - 1)
        embeddings = _call_embeddings_batch(
            texts, model=model, max_retries=max_retries, initial_backoff=initial_backoff, timeout=timeout
        )
        # Attach embeddings back to documents
        for doc, emb in zip(batch, embeddings):
            doc["embedding"] = emb
        LOGGER.debug("Attached embeddings for documents %d-%d", start, end - 1)

    LOGGER.info("Completed embedding %d documents", total)
    return docs


__all__ = ["embed_documents"]
