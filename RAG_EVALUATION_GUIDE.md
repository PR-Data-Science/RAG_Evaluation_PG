# RAG Evaluation System — Quick Start Guide

## 🚀 Running the RAG Evaluation UI

The RAG evaluation UI provides a production-grade system for testing and debugging the Retrieval-Augmented Generation pipeline.

### Start the UI

```bash
cd /Users/pr/Downloads/Learning_Projects/LLM/Agent_UTA_HR_Policies

# Run the evaluation UI (Gradio will launch on http://127.0.0.1:7900)
./.venv/bin/python3 src/ui/rag_evaluation_ui.py
```

The UI will:
1. Run startup tests with 3 sample queries
2. Launch the Gradio interface at `http://127.0.0.1:7900`

### Using the UI

**Input:**
- Enter your HR policy question in the textbox
- Adjust "Top-K Results" (1-15) to retrieve more or fewer chunks

**Output Tabs:**

1. **💡 Generated Answer** — The final answer from OpenAI GPT-4o-mini, grounded in retrieved policy context
2. **📚 Retrieved Context** — The policy excerpts that were used to generate the answer, ranked by FAISS similarity
3. **🔎 Retrieval Table** — Metadata table showing:
   - Rank: Retrieval position (1=most relevant)
   - Distance: FAISS similarity score (0-1, higher is better)
   - Source PDF: Which policy document
   - Page: Page number in the document
4. **📊 Evaluation & Diagnostics** — Detailed metrics:
   - Retrieval coverage (single vs. multi-PDF)
   - Average similarity score
   - Failure warnings and diagnostics

### Example Questions

- "Is a student employee eligible for the Employee Tuition Affordability Program?"
- "What are the eligibility requirements for Family and Medical Leave?"
- "How do I apply for leave?"
- "What policies cover performance evaluations?"

---

## 🏗️ System Architecture

```
User Query
    ↓
[FAISS Retrieval] → Retrieve top-K policy chunks
    ↓
[Context Assembly] → Format chunks with source/page info
    ↓
[OpenAI LLM] → Generate answer grounded in context
    ↓
[Evaluation] → Assess retrieval quality & answer reliability
    ↓
[Logging] → Store failure events in logs/rag_failures.jsonl
```

### Key Modules

- **`src/retrieval/faiss_retriever.py`** — FAISS vector search, deterministic mock embeddings, retrieval diagnostics
- **`src/ui/rag_evaluation_ui.py`** — Gradio interface, RAG pipeline orchestration, evaluation & logging
- **`src/openai_utils.py`** — OpenAI API wrapper with error handling
- **`src/ingestion/pdf_ingestor.py`** — PDF chunking (91 chunks from 12 PDFs)
- **`src/embeddings/embedding_generator.py`** — Embedding generation with batch processing

### Data Files

- **`temp_storage/03_embedded_documents.json`** — All 91 document chunks with metadata
- **`temp_storage/04_metadata_mapping.json`** — Document metadata index
- **`logs/rag_failures.jsonl`** — Append-only log of failure events

---

## 🧪 Testing

Run startup tests without launching the UI:

```bash
./.venv/bin/python3 -c "
import sys
sys.path.insert(0, '.')
from src.ui.rag_evaluation_ui import run_startup_tests
run_startup_tests()
"
```

---

## 📊 Understanding the Output

### Good Retrieval Signals
- ✅ **Multiple PDFs** — Queries span 2+ policy documents
- ✅ **High similarity** — Average distance > 0.15
- ✅ **Clear answer** — LLM provides specific information with citations

### Warning Signs
- ⚠️ **Low similarity** — Distance < 0.15 indicates weak retrieval
- ⚠️ **Single source** — All results from one PDF (narrow coverage)
- ⚠️ **"Not found"** — Answer indicates information not covered in policies
- ⚠️ **No multi-PDF** — Query may not have broad enough policy coverage

---

## 🔧 Advanced Usage

### Custom Prompts

To modify the system prompt used for grounding the LLM, edit the `generate_rag_answer()` function in `src/ui/rag_evaluation_ui.py`.

### Failure Analysis

View logged failures:

```bash
cat logs/rag_failures.jsonl | jq .
```

Each entry includes:
- Timestamp
- Query
- Top retrieved sources
- Average similarity score
- Failure reasons

### Real Embeddings (Future)

To use real OpenAI embeddings instead of mock/deterministic embeddings:
1. Create embeddings via `src/embeddings/embedding_generator.py`
2. Store in FAISS index using `src/storage/faiss_indexer.py`
3. Update `src/retrieval/faiss_retriever.py` to load from disk index

---

## 🐛 Troubleshooting

### "FAISS index file not found"
→ The system automatically builds an in-memory index from document embeddings on first run.

### Low similarity scores
→ This is expected with deterministic mock embeddings. Real OpenAI embeddings will improve scores significantly.

### OpenAI API errors
→ Check that `OPENAI_API_KEY` is set in `.env` file
→ Verify API quota and rate limits

### No retrieval results
→ Try rephrasing the query with HR policy-specific keywords
→ Increase Top-K to retrieve more candidates

---

## 📚 Related Documentation

- [STEP 1: PDF Ingestion](STEP1_INGESTION.md)
- [STEP 2B: FAISS Indexing](STEP2B_FAISS_INDEXING.md)
- [Setup Guide](SETUP_GUIDE.md)

---

**Status**: ✅ Full RAG system working end-to-end with real OpenAI generation

Last updated: December 22, 2025
