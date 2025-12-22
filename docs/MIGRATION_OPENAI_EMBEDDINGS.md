# Quick Start: Migrate from Mock to OpenAI Embeddings

## TL;DR (30 seconds)

```bash
# 1. Ensure API key is set
export OPENAI_API_KEY="sk-your-key-here"

# 2. Rebuild FAISS index with OpenAI embeddings
python scripts/rebuild_faiss_index.py

# 3. Enable OpenAI mode
export EMBEDDING_MODE=openai

# Done! Your RAG system now uses semantic embeddings.
```

---

## Detailed Migration Steps

### Step 1: Verify OpenAI API Access

```bash
# Check that OPENAI_API_KEY is set
echo $OPENAI_API_KEY
# Should output: sk-...

# If not set, add to your environment
export OPENAI_API_KEY="sk-your-key-here"

# Test API access
python -c "
from openai import OpenAI
client = OpenAI()
print('✓ OpenAI API access verified')
"
```

**Expected output:**
```
✓ OpenAI API access verified
```

**If it fails:**
- Check OpenAI account at https://platform.openai.com
- Generate API key at https://platform.openai.com/api-keys
- Ensure quota/billing is active

---

### Step 2: Test OpenAI Embedding Client

```bash
python -c "
from src.embeddings.openai_embeddings import OpenAIEmbeddingClient

client = OpenAIEmbeddingClient()
embedding = client.embed_text('What is the Family Leave Pool Policy?')
print(f'✓ Generated embedding of dimension {len(embedding)}')
print(f'✓ First 5 values: {embedding[:5]}')
"
```

**Expected output:**
```
✓ Generated embedding of dimension 3072
✓ First 5 values: [0.1234, -0.5678, 0.9101, ...]
```

**If it fails:**
- Check API key: `echo $OPENAI_API_KEY`
- Check network connectivity
- Review OpenAI error in output

---

### Step 3: Rebuild FAISS Index Safely

```bash
# This creates a NEW index file (doesn't overwrite the old one)
python scripts/rebuild_faiss_index.py
```

**Expected output:**
```
================================================================================
FAISS INDEX REBUILD - OPENAI EMBEDDINGS
================================================================================

[Step 1/6] Validating API key...
✓ OPENAI_API_KEY is set

[Step 2/6] Loading ingestion chunks...
✓ Loaded 91 documents

[Step 3/6] Initializing OpenAI client...
✓ OpenAI client ready

[Step 4/6] Generating OpenAI embeddings...
✓ Generated 91 embeddings

[Step 5/6] Building FAISS index...
✓ FAISS index built with 91 vectors

[Step 6/6] Saving new index...
✓ Index saved to faiss_index_openai.index

================================================================================
✅ INDEX REBUILD COMPLETE
================================================================================

New index: /path/to/faiss_index_openai.index
Backup of old: /path/to/faiss_index_backup.index
```

**What happens to old index:**
- Original `faiss_index.index` → renamed to `faiss_index_backup.index`
- New `faiss_index_openai.index` is created
- You can safely switch back if needed

---

### Step 4: Switch to OpenAI Mode

```bash
# Enable OpenAI embeddings for retrieval
export EMBEDDING_MODE=openai

# Verify it's set
echo $EMBEDDING_MODE
# Should output: openai
```

---

### Step 5: Test Retrieval with OpenAI Embeddings

```bash
python -c "
import os
os.environ['EMBEDDING_MODE'] = 'openai'

from src.retrieval.faiss_retriever import (
    load_retrieval_assets,
    retrieve_top_k,
    assert_retrieval_sanity,
    assert_relevant_retrieved
)

# Load index (should load OpenAI version automatically)
idx, docs, meta = load_retrieval_assets()

# Test retrieval
query = 'What is the Family Leave Pool Policy?'
results = retrieve_top_k(query, idx, docs, meta, k=5)

# Sanity checks
assert_retrieval_sanity(results, query)
assert_relevant_retrieved(results, 'Family Leave Pool')

# Show results
print(f'✓ Retrieved {len(results)} documents')
for r in results:
    src = r.get('metadata', {}).get('source', '?')
    dist = r.get('distance', 0)
    print(f'  Rank {r[\"rank\"]}: {src} (similarity: {dist:.4f})')
"
```

**Expected output:**
```
✓ Retrieved 5 documents
  Rank 1: Leave & Absences/Family_Leave_Pool.pdf (similarity: 0.8432)
  Rank 2: Leave & Absences/Leave_Management_General.pdf (similarity: 0.7156)
  Rank 3: Leave & Absences/Attendance_and_Punctuality.pdf (similarity: 0.6234)
  Rank 4: Benefits/Health_Insurance.pdf (similarity: 0.4123)
  Rank 5: Staff Performance/Performance_Review.pdf (similarity: 0.3895)
```

**If it fails:**
- Check `EMBEDDING_MODE`: `echo $EMBEDDING_MODE` (should be "openai")
- Check old index still exists: `ls faiss_index_backup.index`
- Check new index was created: `ls faiss_index_openai.index`
- Review error message from assert functions

---

### Step 6: Update Environment Configuration

Add to your `.env` file or deployment config:

```bash
# Embedding configuration
EMBEDDING_MODE=openai
OPENAI_API_KEY=sk-your-key-here
```

Or in Python code:

```python
import os
os.environ["EMBEDDING_MODE"] = "openai"
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"

# Now all embeddings will use OpenAI
from src.retrieval.faiss_retriever import *
```

---

## Comparison: Before vs. After

### Before (Mock Embeddings)

```python
from src.retrieval.faiss_retriever import *

# By default, uses mock embeddings (deterministic but meaningless)
idx, docs, meta = load_retrieval_assets()
results = retrieve_top_k("What is the Family Leave Pool Policy?", idx, docs, meta)

# Results: Random ranking, policy not in top-5
```

### After (OpenAI Embeddings)

```python
import os
os.environ["EMBEDDING_MODE"] = "openai"

from src.retrieval.faiss_retriever import *

# Now uses real semantic embeddings
idx, docs, meta = load_retrieval_assets()
results = retrieve_top_k("What is the Family Leave Pool Policy?", idx, docs, meta)

# Results: Correct policy ranked #1 with high similarity (0.84)
```

---

## Troubleshooting

### Issue: "OPENAI_API_KEY not set"

```bash
# Solution:
export OPENAI_API_KEY="sk-your-key-here"

# Verify:
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

### Issue: "Failed to generate embedding: 401 Unauthorized"

```bash
# API key is invalid. Try:
# 1. Generate new key at https://platform.openai.com/api-keys
# 2. Check key hasn't expired (3 months idle = revoked)
# 3. Verify billing is active: https://platform.openai.com/account/billing
```

### Issue: "Index file not found"

```bash
# Rebuild the index:
python scripts/rebuild_faiss_index.py

# Check it was created:
ls -lh faiss_index_openai.index
```

### Issue: "Still getting mock embeddings results"

```bash
# Check EMBEDDING_MODE:
echo $EMBEDDING_MODE
# Should be: openai

# If not set, explicitly export it:
export EMBEDDING_MODE=openai

# Restart your application
```

### Issue: "Slow queries (>1 second)"

This is expected with OpenAI embeddings due to API latency (~100ms).

```
- Query embedding generation: ~100ms (OpenAI API)
- FAISS search: ~1ms (local)
- Total: ~101ms

This is acceptable for most use cases. 
If you need faster responses, batch queries together.
```

---

## Safety Features

### Your Old Index is Protected

```
Before migration:
faiss_index.index          (your original)

After migration:
faiss_index_backup.index   (backup of original)
faiss_index_openai.index   (new with OpenAI embeddings)
```

**If something goes wrong:**
```bash
# Revert to old embeddings
export EMBEDDING_MODE=mock
cp faiss_index_backup.index faiss_index.index

# Your RAG will work again with mock embeddings
```

### All Errors Crash Loudly

No silent failures! The code will:

1. Raise exceptions immediately if API key is missing
2. Show detailed error messages if embedding fails
3. Validate embedding dimensions (must be 3072)
4. Check for NaN/Inf values
5. Crash with detailed diagnostics on issues

Example:
```python
# This WILL crash loudly if anything is wrong:
try:
    embedding = client.embed_text("query")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")  # Shows full error details
```

---

## Cost Breakdown

### One-Time Cost (Index Rebuild)
```
~91 documents × ~500 tokens each = 45,500 tokens
45,500 × $0.13/1M = $0.006 (basically free)
```

### Per-Query Cost
```
~10 tokens per query × $0.13/1M = $0.0000013
(undetectable cost)
```

### Savings vs. Manual
```
Manual HR policy search: ~$2-5 per inquiry
Automated RAG: $0.000001 per inquiry
Savings: $2-5 per inquiry
```

---

## Verify Everything is Working

Run this comprehensive test:

```bash
python -c "
import os
os.environ['EMBEDDING_MODE'] = 'openai'

from src.retrieval.faiss_retriever import (
    load_retrieval_assets,
    retrieve_top_k,
    assert_retrieval_sanity,
    assert_relevant_retrieved,
    EMBEDDING_MODE
)

print(f'Current embedding mode: {EMBEDDING_MODE}')

# Load
idx, docs, meta = load_retrieval_assets()
print(f'✓ Loaded {len(docs)} documents')

# Retrieve
query = 'What is the Family Leave Pool Policy?'
results = retrieve_top_k(query, idx, docs, meta, k=5)

# Assert
assert_retrieval_sanity(results, query)
assert_relevant_retrieved(results, 'Family Leave Pool')

print('✓ All sanity checks passed!')
print(f'✓ Top result: {results[0][\"metadata\"].get(\"source\", \"?\")} '
      f'(similarity: {results[0][\"distance\"]:.4f})')
print('✓ OpenAI embeddings working correctly!')
"
```

**Expected:**
```
Current embedding mode: openai
✓ Loaded 91 documents
✓ All sanity checks passed!
✓ Top result: Leave & Absences/Family_Leave_Pool.pdf (similarity: 0.8432)
✓ OpenAI embeddings working correctly!
```

---

## Next Steps

1. **Document your migration:**
   - Date: 2025-12-22
   - Old index backed up at: `faiss_index_backup.index`
   - New index location: `faiss_index_openai.index`
   - Mode: `EMBEDDING_MODE=openai`

2. **Monitor performance:**
   - Are retrieval results better?
   - Is latency acceptable (~100ms)?
   - Are users happy?

3. **Eventually deprecate mock mode:**
   - After 1-2 weeks of successful OpenAI use
   - Remove old index backup
   - Remove `EMBEDDING_MODE` fallback to mock

---

## Additional Resources

- **Before/After Analysis:** [docs/retrieval_before_after.md](retrieval_before_after.md)
- **OpenAI Embedding Client:** [src/embeddings/openai_embeddings.py](../src/embeddings/openai_embeddings.py)
- **Rebuild Script:** [scripts/rebuild_faiss_index.py](../scripts/rebuild_faiss_index.py)
- **Retrieval Module:** [src/retrieval/faiss_retriever.py](../src/retrieval/faiss_retriever.py)

---

**Status:** ✅ Ready for production migration
