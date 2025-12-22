#!/usr/bin/env python
"""
Full pipeline: Ingestion + Embedding → Temporary Storage
Runs STEP 1 and STEP 2A together, storing results for verification.
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')

print("=" * 80)
print("FULL PIPELINE: Ingestion → Embedding → Temporary Storage")
print("=" * 80)

# Create temp storage directory
temp_dir = Path("temp_storage")
temp_dir.mkdir(exist_ok=True)
print(f"\n✓ Created temp storage: {temp_dir}\n")

# STEP 1: PDF Ingestion
print("[STEP 1] Running PDF Ingestion...")
from src.ingestion.pdf_ingestor import PDFIngestor

ingestor = PDFIngestor()
chunks = ingestor.load_pdfs("DataSources/UTA_HR_policies")
print(f"✓ Loaded {len(chunks)} chunks\n")

# Save chunks to temp storage
chunks_file = temp_dir / "01_ingestion_chunks.json"
with open(chunks_file, "w") as f:
    chunks_serializable = []
    for chunk in chunks:
        chunks_serializable.append({
            "id": chunk["id"],
            "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
            "metadata": chunk["metadata"]
        })
    json.dump(chunks_serializable, f, indent=2, default=str)

print(f"✓ Saved chunks summary to: {chunks_file}\n")

# STEP 2: Prepare for embedding
print("[STEP 2] Preparing documents for embedding...")
docs_for_embedding = []
for chunk in chunks:
    docs_for_embedding.append({
        "id": chunk["id"],
        "text": chunk["content"],
        "metadata": chunk["metadata"]
    })

print(f"✓ Prepared {len(docs_for_embedding)} documents\n")

# STEP 2A: Generate embeddings
print("[STEP 2A] Generating embeddings...")
if os.getenv("OPENAI_API_KEY"):
    print("  Using OpenAI real embeddings...")
    try:
        from src.embeddings.embedding_generator import embed_documents
        embedded_docs = embed_documents(docs_for_embedding, batch_size=50)
        print(f"✓ Generated real embeddings for {len(embedded_docs)} documents")
        embedding_source = "OpenAI (real)"
    except Exception as e:
        print(f"⚠️  Real embeddings failed: {e}")
        print("  Falling back to mock embeddings...")
        for doc in docs_for_embedding:
            doc["embedding"] = np.random.randn(3072).tolist()
        embedded_docs = docs_for_embedding
        embedding_source = "Mock (random)"
else:
    print("  OPENAI_API_KEY not set, using mock embeddings...")
    for doc in docs_for_embedding:
        doc["embedding"] = np.random.randn(3072).tolist()
    embedded_docs = docs_for_embedding
    embedding_source = "Mock (random)"

print(f"✓ Embeddings source: {embedding_source}\n")

# Save embedding statistics
embedding_stats = {
    "total_documents": len(embedded_docs),
    "embedding_dimension": 3072,
    "embedding_source": embedding_source,
    "sample_embedding": {
        "doc_id": embedded_docs[0]["id"],
        "embedding_length": len(embedded_docs[0]["embedding"]),
        "first_10_values": embedded_docs[0]["embedding"][:10]
    }
}

stats_file = temp_dir / "02_embedding_stats.json"
with open(stats_file, "w") as f:
    json.dump(embedding_stats, f, indent=2)

print(f"✓ Saved embedding stats to: {stats_file}\n")

# STEP 3: Save full documents with embeddings
print("[STEP 3] Storing documents with embeddings...")
full_docs_file = temp_dir / "03_embedded_documents.json"

docs_summary = []
for doc in embedded_docs:
    docs_summary.append({
        "id": doc["id"],
        "text_preview": doc["text"][:150] + "..." if len(doc["text"]) > 150 else doc["text"],
        "text_length": len(doc["text"]),
        "embedding_length": len(doc["embedding"]),
        "metadata": doc["metadata"]
    })

with open(full_docs_file, "w") as f:
    json.dump(docs_summary, f, indent=2, default=str)

print(f"✓ Saved {len(docs_summary)} documents to: {full_docs_file}\n")

# STEP 4: Create metadata mapping
print("[STEP 4] Creating metadata mapping...")
metadata_mapping = {}
for doc in embedded_docs:
    metadata_mapping[doc["id"]] = {
        "source": doc["metadata"]["source"],
        "page": doc["metadata"]["page"],
        "chunk_index": doc["metadata"]["chunk_index"],
        "chunk_size": doc["metadata"]["chunk_size"],
        "total_chunks": doc["metadata"]["total_chunks"],
        "text_length": len(doc["text"])
    }

metadata_file = temp_dir / "04_metadata_mapping.json"
with open(metadata_file, "w") as f:
    json.dump(metadata_mapping, f, indent=2)

print(f"✓ Saved metadata for {len(metadata_mapping)} documents to: {metadata_file}\n")

# STEP 5: Generate summary report
print("[STEP 5] Generating summary report...")
by_source = {}
for doc_id, meta in metadata_mapping.items():
    source = meta["source"]
    if source not in by_source:
        by_source[source] = []
    by_source[source].append(doc_id)

summary_report = {
    "pipeline_status": "SUCCESS",
    "total_chunks": len(embedded_docs),
    "total_pdfs": len(by_source),
    "embedding_dimension": 3072,
    "embedding_source": embedding_source,
    "files_by_pdf": {source: len(doc_ids) for source, doc_ids in by_source.items()},
    "stored_files": {
        "chunks": str(chunks_file),
        "embedding_stats": str(stats_file),
        "documents": str(full_docs_file),
        "metadata": str(metadata_file)
    }
}

report_file = temp_dir / "PIPELINE_REPORT.json"
with open(report_file, "w") as f:
    json.dump(summary_report, f, indent=2)

print(f"✓ Saved pipeline report to: {report_file}\n")

# Display summary
print("=" * 80)
print("PIPELINE EXECUTION SUMMARY")
print("=" * 80)
print(f"\n📁 Temporary Storage: {temp_dir.absolute()}\n")

print(f"📄 Stored Files:")
print(f"  1. {chunks_file.name}")
print(f"     └─ {len(chunks)} chunks with metadata")
print(f"  2. {stats_file.name}")
print(f"     └─ Embedding statistics and sample")
print(f"  3. {full_docs_file.name}")
print(f"     └─ All {len(embedded_docs)} documents with embeddings summary")
print(f"  4. {metadata_file.name}")
print(f"     └─ Complete metadata mapping")
print(f"  5. PIPELINE_REPORT.json")
print(f"     └─ Full pipeline execution report")

print(f"\n📊 Pipeline Results:")
print(f"  ✓ Total Chunks: {len(embedded_docs)}")
print(f"  ✓ Total PDFs: {len(by_source)}")
print(f"  ✓ Embedding Dimension: 3072")
print(f"  ✓ Embedding Source: {embedding_source}")

print(f"\n📋 Breakdown by PDF:")
for source in sorted(by_source.keys()):
    count = len(by_source[source])
    print(f"  ✓ {source}: {count} chunks")

print("\n" + "=" * 80)
print("✅ PIPELINE COMPLETE - READY FOR VERIFICATION")
print("=" * 80)
