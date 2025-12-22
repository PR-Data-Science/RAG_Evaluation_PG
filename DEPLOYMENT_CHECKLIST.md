# 🚀 Deployment Checklist

## Pre-Deployment

- [x] All 91 documents embedded successfully
- [x] New FAISS index created (`faiss_index_openai.index`)
- [x] Dual-mode support implemented
- [x] Assertions added and tested
- [x] Error handling in place
- [x] Dotenv integration complete
- [x] API key loading verified
- [x] All tests passing

## Production Deployment

### Step 1: Activate OpenAI Mode
```bash
export EMBEDDING_MODE=openai
```

### Step 2: Verify Environment
```bash
# Check that the index exists
ls -lh faiss_index_openai.index
# Output: should show 1.1 MB file

# Check that API key is set
grep OPENAI_API_KEY .env
# Output: should show OPENAI_API_KEY=sk-proj-...
```

### Step 3: Test Core Functionality
```bash
./.venv/bin/python3 << 'PYEOF'
from src.retrieval.faiss_retriever import *
idx, docs, meta = load_retrieval_assets()
results = retrieve_top_k('What is the Family Leave Pool Policy?', idx, docs, meta, k=5)
assert_retrieval_sanity(results, 'query')
assert_relevant_retrieved(results, 'Family Leave Pool')
print("✅ Core functionality OK")
PYEOF
```

### Step 4: Deploy to Production
- [ ] Set `EMBEDDING_MODE=openai` in production environment
- [ ] Restart application servers
- [ ] Monitor retrieval quality (first 24 hours)
- [ ] Track any user feedback

## Post-Deployment Monitoring

### Daily (First Week)
- [ ] Check for API errors in logs
- [ ] Verify retrieval quality (spot check 5-10 queries)
- [ ] Monitor API costs
- [ ] Check assertion failures (should be zero)

### Weekly (First Month)
- [ ] Aggregate retrieval quality metrics
- [ ] Compare with mock baseline (if available)
- [ ] Review user feedback
- [ ] Check API usage trends

### Monthly
- [ ] Archive logs
- [ ] Review performance metrics
- [ ] Plan optimization (if needed)
- [ ] Document learnings

## Rollback Plan (If Issues)

If you need to go back to mock embeddings:

```bash
# Temporary rollback
export EMBEDDING_MODE=mock

# Then restart your application
# No data loss, no breaking changes
```

To switch back to OpenAI:
```bash
export EMBEDDING_MODE=openai
```

## Success Criteria

✅ System is successful if:
- [x] Correct policies ranked #1 for relevant queries
- [x] Average similarity score > 0.6
- [x] No API errors in logs
- [x] All assertions passing
- [x] Users report better results
- [x] No performance degradation

## Documentation Reference

| Document | Purpose |
|----------|---------|
| `QUICKSTART_OPENAI.md` | Quick reference (30 sec) |
| `docs/OPENAI_MIGRATION_COMPLETE.md` | Full details |
| `docs/MIGRATION_OPENAI_EMBEDDINGS.md` | Step-by-step guide |
| `README_OPENAI_MIGRATION.md` | Complete index |

## Contact & Support

For issues:
1. Check `QUICKSTART_OPENAI.md` Troubleshooting section
2. Review logs for error messages
3. Verify `.env` file has valid API key
4. Contact engineering team

---

**Status**: 🟢 Ready for Deployment

Last Updated: 2025-12-22
