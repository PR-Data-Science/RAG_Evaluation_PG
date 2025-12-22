from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np


INDEX_FILENAME_CANDIDATES = [Path("faiss_index.index"), Path("faiss_index") / "faiss_index.index"]
DOCUMENTS_PATH = Path("temp_storage") / "03_embedded_documents.json"
METADATA_PATH = Path("temp_storage") / "04_metadata_mapping.json"
EMBED_DIM = 3072


def load_retrieval_assets(index_path: Path | None = None) -> Tuple[faiss.Index, List[Dict[str, Any]], Dict[str, Any]]:
    """Load FAISS index, embedded documents and metadata mapping.

    Args:
        index_path: Optional path to the FAISS index file. If None, tries sensible defaults.

    Returns:
        (index, documents, metadata_map)

    Raises:
        FileNotFoundError: if any required file is missing.
    """
    # Resolve index path (optional)
    resolved: Path | None = None
    if index_path is None:
        for candidate in INDEX_FILENAME_CANDIDATES:
            if candidate.exists():
                resolved = candidate
                break
    else:
        resolved = index_path

    # Load documents and metadata first (we may build an index from embeddings)
    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(f"Embedded documents JSON not found at {DOCUMENTS_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata mapping JSON not found at {METADATA_PATH}")

    print(f"Loading embedded documents from: {DOCUMENTS_PATH}")
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Loading metadata mapping from: {METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata_map = json.load(f)

    # If FAISS index file exists, read it. Otherwise build an in-memory index from document embeddings.
    if resolved is not None and resolved.exists():
        print(f"Loading FAISS index from: {resolved}")
        index = faiss.read_index(str(resolved))
    else:
        print("FAISS index file not found on disk — building in-memory FAISS index from embedded documents' embeddings.")
        # Extract embeddings from documents (generate deterministic mock embeddings when missing)
        vecs = []
        for i, doc in enumerate(documents):
            emb = doc.get("embedding")
            if emb is None:
                # Create a deterministic mock embedding based on doc id/text so retrieval works for learning/testing
                print(f"Document {i} missing 'embedding' — generating deterministic mock embedding for document id={doc.get('id')}")
                emb_vec = embed_from_document(doc)
            else:
                emb_vec = np.array(emb, dtype=np.float32)
                if emb_vec.shape[0] != EMBED_DIM:
                    raise ValueError(f"Embedding for document {doc.get('id')} has incorrect dimension {emb_vec.shape}")
            vecs.append(emb_vec)

        xb = np.vstack(vecs).astype(np.float32)
        # Normalize to unit length for cosine-like inner product
        faiss.normalize_L2(xb)

        index = faiss.IndexFlatIP(EMBED_DIM)
        index.add(xb)

    # Basic consistency check
    if hasattr(index, "ntotal"):
        idx_count = int(index.ntotal)
        doc_count = len(documents)
        print(f"FAISS index vectors: {idx_count}, documents: {doc_count}")
        if idx_count != doc_count:
            print("WARNING: index vector count != number of documents. Retrieval ordering may not match documents list.")

    return index, documents, metadata_map


def embed_query(query: str) -> np.ndarray:
    """Create a deterministic, mock embedding for `query`.

    This currently uses a query-hash-derived RNG seed to produce a reproducible
    random vector of length `EMBED_DIM`.

    NOTE: This will be replaced with OpenAI embeddings later.
    """
    # Deterministic seed from query hash
    h = hashlib.sha256(query.encode("utf-8")).hexdigest()
    seed = int(h[:16], 16) % (2**31 - 1)
    rng = np.random.default_rng(seed)

    vec = rng.standard_normal(EMBED_DIM, dtype=np.float32)
    # Normalize to unit length because FAISS index was built on normalized vectors
    vec = vec.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(vec)
    return vec[0]


def embed_from_document(doc: Dict[str, Any]) -> np.ndarray:
    """Create a deterministic mock embedding for a document.

    Uses the document `id` or `text_preview` (fallback) as the basis
    for a reproducible embedding. This allows building an index when
    the stored embedding arrays are not present (common in learning/demo runs).
    """
    seed_source = doc.get("id") or doc.get("text_preview") or json.dumps(doc, sort_keys=True)
    h = hashlib.sha256(str(seed_source).encode("utf-8")).hexdigest()
    seed = int(h[:16], 16) % (2**31 - 1)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(EMBED_DIM, dtype=np.float32)
    vec = vec.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(vec)
    return vec[0]


def _get_text_from_doc(doc: Dict[str, Any]) -> str:
    # Accept multiple possible text keys for flexibility
    for key in ("text", "content", "text_preview", "chunk_text"):
        if key in doc:
            return doc[key]
    # fallback: try to join any string fields
    for v in doc.values():
        if isinstance(v, str) and len(v) > 20:
            return v
    return ""


def retrieve_top_k(
    query: str,
    index: faiss.Index,
    documents: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve top-k documents for `query` using FAISS.

    Returns a ranked list of dicts with keys: `rank`, `distance`, `document_id`,
    `text_preview`, `metadata`.
    """
    if index is None:
        raise ValueError("index is None")
    if not isinstance(documents, list):
        raise ValueError("documents must be a list in the same order used to build the FAISS index")

    q_emb = embed_query(query)
    q_arr = q_emb.reshape(1, -1).astype(np.float32)

    # FAISS expects 2D array
    # Search
    topk = min(k, int(index.ntotal)) if hasattr(index, "ntotal") else k
    distances, indices = index.search(q_arr, topk)
    distances = distances[0].tolist()
    indices = indices[0].tolist()

    results: List[Dict[str, Any]] = []
    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        if idx < 0:
            continue
        try:
            doc = documents[idx]
        except Exception:
            # If index ordering doesn't match, best-effort try to look up by id inside metadata
            doc = None

        if doc is None:
            doc_id = None
            doc_meta = None
            text_preview = ""
        else:
            doc_id = doc.get("id")
            doc_meta = doc.get("metadata") or metadata.get(doc_id)
            full_text = _get_text_from_doc(doc)
            text_preview = full_text[:300]

        results.append(
            {
                "rank": rank,
                "distance": float(dist),
                "document_id": doc_id,
                "text_preview": text_preview,
                "metadata": doc_meta,
            }
        )

    return results


def audit_retrieval(results: List[Dict[str, Any]]) -> None:
    """Print diagnostics about retrieval results.

    - Shows number of retrieved chunks per source PDF.
    - Warns if all results come from the same document.
    - Warns if average distances indicate low similarity.

    Reasons retrieval can fail (examples):
    - Bad embeddings: embeddings do not capture the semantic meaning (mock/random vectors).
    - Mismatched normalization: query vector not normalized while index vectors are (we normalize here).
    - Chunking granularity: chunks too small/large causing the relevant text to be split or diluted.
    - Metadata/index ordering mismatch: index vector order doesn't match documents list order.
    - Missing documents or corrupted index file.
    """
    if not results:
        print("No results to audit.")
        return

    # Count by source
    source_counts: Dict[str, int] = {}
    doc_ids = set()
    distances: List[float] = []
    for r in results:
        meta = r.get("metadata") or {}
        source = meta.get("source") if isinstance(meta, dict) else None
        if source is None:
            source = "<unknown>"
        source_counts[source] = source_counts.get(source, 0) + 1
        if r.get("document_id"):
            doc_ids.add(r["document_id"])
        distances.append(r.get("distance", 0.0))

    print("\n=== Retrieval Audit ===")
    print("Retrieved chunks per source:")
    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt}")

    if len(doc_ids) == 1:
        print("WARNING: All retrieved chunks come from the same document. This may indicate poor embedding diversity or overly-specific query/chunking.")

    avg_dist = float(np.mean(distances)) if distances else 0.0
    print(f"Average distance (similarity score): {avg_dist:.4f}")
    # For normalized inner product (cosine-like), values closer to 1 mean high similarity.
    if avg_dist < 0.20:
        print("WARNING: Average similarity is low — results may not be relevant. Possible causes: mock embeddings, wrong normalization, or missing context.")
    elif avg_dist < 0.5:
        print("NOTE: Similarity is moderate. Consider increasing number of retrieved chunks or improving embeddings.")


if __name__ == "__main__":
    # Example execution
    try:
        idx, docs, meta = load_retrieval_assets()
    except Exception as e:
        print("Failed to load retrieval assets:", e)
        raise

    query = "Is a student employee eligible for the Employee Tuition Affordability Program?"
    print(f"\nRunning retrieval for query: {query}\n")
    res = retrieve_top_k(query, idx, docs, meta, k=5)

    print("\nTop results:")
    for r in res:
        src = (r.get("metadata") or {}).get("source") if r.get("metadata") else "<unknown>"
        print(f"Rank {r['rank']}: distance={r['distance']:.4f} source={src}")

    audit_retrieval(res)
