#!/usr/bin/env python3
"""Rebuild embeddings from ingestion chunks and create FAISS index.

Steps:
- Load `temp_storage/01_ingestion_chunks.json`
- Convert to docs with keys `id`,`text`,`metadata`
- Call `src.embeddings.embedding_generator.embed_documents`
- Save embedded docs to `temp_storage/03_embedded_documents.json`
- Build FAISS index into `faiss_index` dir and copy to `faiss_index_openai.index`
"""
from __future__ import annotations
import json
import os
from pathlib import Path

from src.embeddings.embedding_generator import embed_documents
from src.storage.faiss_indexer import build_faiss_index

ROOT = Path(__file__).resolve().parents[1]
INGEST_PATH = ROOT / "temp_storage" / "01_ingestion_chunks.json"
EMBEDDED_OUT = ROOT / "temp_storage" / "03_embedded_documents.json"
INDEX_DIR = ROOT / "faiss_index"
ROOT_INDEX_FILE = ROOT / "faiss_index_openai.index"


def main():
    if not INGEST_PATH.exists():
        raise FileNotFoundError(f"Ingestion chunks not found at {INGEST_PATH}")

    with open(INGEST_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    docs = []
    for c in chunks:
        doc = {
            "id": c.get("id"),
            "text": c.get("content") or c.get("text") or c.get("text_preview") or "",
            "metadata": c.get("metadata", {}),
        }
        docs.append(doc)

    print(f"Generating embeddings for {len(docs)} documents...")
    docs_with_emb = embed_documents(docs, batch_size=64)

    # Prepare output format similar to existing 03_embedded_documents.json
    out_docs = []
    for d in docs_with_emb:
        emb = d.get("embedding")
        out_docs.append(
            {
                "id": d.get("id"),
                "text_preview": (d.get("text")[:400] if d.get("text") else ""),
                "text_length": len(d.get("text") or ""),
                "embedding_length": len(emb) if emb else 0,
                "embedding": emb,
                "metadata": d.get("metadata", {}),
            }
        )

    EMBEDDED_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDED_OUT, "w", encoding="utf-8") as fh:
        json.dump(out_docs, fh, ensure_ascii=False, indent=2)

    print("Saved embedded documents to", EMBEDDED_OUT)

    # Build FAISS index using storage module
    # build_faiss_index expects docs with 'id','text','embedding','metadata'
    build_docs = []
    for d in docs_with_emb:
        build_docs.append({
            "id": d.get("id"),
            "text": d.get("text"),
            "embedding": d.get("embedding"),
            "metadata": d.get("metadata", {}),
        })

    print(f"Building FAISS index in {INDEX_DIR} ...")
    build_faiss_index(build_docs, str(INDEX_DIR))

    # Copy built index file to root as faiss_index_openai.index for compatibility
    src_index = INDEX_DIR / "faiss.index"
    if src_index.exists():
        with open(src_index, "rb") as r, open(ROOT_INDEX_FILE, "wb") as w:
            w.write(r.read())
        print("Wrote root index file:", ROOT_INDEX_FILE)
    else:
        print("Warning: expected index file not found at", src_index)


if __name__ == "__main__":
    main()
