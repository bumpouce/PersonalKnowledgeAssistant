# Personal Knowledge Assistant - Quick Reference

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                               │
├──────────────────────────┬──────────────────────────────────────────┤
│   Web UI (React)         │        CLI Tool (Typer)                  │
│  - Upload documents      │    - Test ingestion/search               │
│  - Chat interface        │    - Development convenience             │
│  - Browse documents      │                                           │
└──────────────────────────┴──────────────────────────────────────────┘
                                    │
                        ┌───────────┴──────────────┐
                        │                          │
                   FastAPI Backend                 │
                   (uvicorn)                       │
                        │                          │
                ┌───────┴──────────┐               │
                │                  │               │
          ┌─────▼─────┐      ┌────▼───────┐       │
          │  Upload   │      │   Search   │       │
          │ /ingest   │      │  /search   │       │
          └─────┬─────┘      └────┬───────┘       │
                │                  │               │
          ┌─────▼──────────────────▼────────┐     │
          │                                  │     │
          │    Ingestion Pipeline           │     │
          │  ┌──────────────────────────┐  │     │
          │  │ 1. Load Document         │  │     │
          │  │    (PDF, TXT, URL)       │  │     │
          │  └──────────┬───────────────┘  │     │
          │             │                   │     │
          │  ┌──────────▼───────────────┐  │     │
          │  │ 2. Split into Chunks     │  │     │
          │  │    (512 tokens, 100 ovlp)│  │     │
          │  └──────────┬───────────────┘  │     │
          │             │                   │     │
          │  ┌──────────▼───────────────┐  │     │
          │  │ 3. Generate Embeddings   │  │     │
          │  │    (HuggingFace)         │  │     │
          │  └──────────┬───────────────┘  │     │
          │             │                   │     │
          │  ┌──────────▼───────────────┐  │     │
          │  │ 4. Store in Pinecone     │  │     │
          │  │    + Metadata            │  │     │
          │  └──────────────────────────┘  │     │
          │                                  │     │
          └──────────────────────────────────┘     │
                                                   │
                    ┌──────────────────────────────┴────────┐
                    │                                       │
            ┌───────▼─────────────────┐        ┌──────────▼──────┐
            │  HuggingFace Models     │        │  Pinecone VectorDB
            │  ─────────────────────  │        │  ────────────────
            │ • all-MiniLM-L6-v2      │        │ • Store vectors
            │ • Runs locally or remote│        │ • Semantic search
            │ • 384-dim embeddings    │        │ • Metadata filter
            │ • <1 sec per chunk      │        │ • Cosine similarity
            └─────────────────────────┘        │ • Serverless scale
                    △                           └────────────────┘
                    │                                  △
                    │                                  │
            ┌───────┴──────────────────────────────────┘
            │
            │    Retrieval Pipeline
            │  ┌────────────────────────────┐
            │  │ 1. User asks: "What is X?" │
            │  └────────────┬─────────────────┘
            │               │
            │  ┌────────────▼─────────────────┐
            │  │ 2. Embed question           │
            │  │    (same model as docs)     │
            │  └────────────┬─────────────────┘
            │               │
            │  ┌────────────▼─────────────────┐
            │  │ 3. Vector similarity search │
            │  │    Get top-k matches        │
            │  └────────────┬─────────────────┘
            │               │
            │  ┌────────────▼─────────────────┐
            │  │ 4. Retrieve full context    │
            │  │    + metadata + scores      │
            │  └────────────────────────────┘
            │
            └───────────────────────────────────────────┐
                                                        │
                        ┌───────────────────────────────┘
                        │
              ┌─────────▼──────────┐
              │  Optional: LLM     │
              │  Integration       │
              │  ───────────────   │
              │ • OpenAI GPT-4/3.5 │
              │ • Context assembly │
              │ • Answer generation│
              │ • Source citing    │
              └────────────────────┘
```

---

## Technology Comparison Table

### Embedding Models (HuggingFace)

| Model | Speed | Quality | Size | Use Case |
|-------|-------|---------|------|----------|
| all-MiniLM-L6-v2 | ⚡⚡⚡ | ⭐⭐⭐ | 22M | MVP, Testing |
| all-mpnet-base-v2 | ⚡⚡ | ⭐⭐⭐⭐ | 109M | Production |
| bge-large-en-v1.5 | ⚡ | ⭐⭐⭐⭐⭐ | 335M | Premium Quality |

### Vector Database Options (Not just Pinecone)

| DB | Cost | Setup | Features | Best For |
|----|------|-------|----------|----------|
| Pinecone | $0-1000/mo | Serverless | Managed, scales easily | Getting started |
| Weaviate | Free-$2000/mo | Self-hosted | Open-source, flexible | Control & customization |
| Milvus | Free | Self-hosted | High-perf, distributed | Large-scale projects |
| Chroma | Free | Local/Cloud | Simple, fast | Learning & small projects |

### LLM Options

| Provider | Cost | Quality | Setup | Use Case |
|----------|------|---------|-------|----------|
| OpenAI | $0.003/1K tokens | Best | API key | Professional quality |
| HuggingFace Inference | Free tier | Good | API key | Learning & experiments |
| Ollama (local) | Free | Good | Local install | Private, no costs |
| Anthropic Claude | $0.003/1K tokens | Best | API key | Long context |

---

## Key Metrics & Optimization Tips

### Embedding Quality
```
Good embeddings = Similar texts have high cosine similarity (>0.7)
Bad embeddings = Random similarity scores

Check: Ask a question and manually verify top results are relevant
```

### Search Relevance
```
Metric: Mean Reciprocal Rank (MRR)
- If correct answer is #1: MRR = 1.0 (perfect)
- If correct answer is #5: MRR = 0.2 (poor)

Target: MRR > 0.7 with top-5 results
```

### Performance Targets
```
Ingestion: < 5 sec per document
Search: < 500ms for query response
End-to-end: < 2 seconds with LLM
```

### Cost Breakdown (Monthly Estimate)
```
Pinecone (serverless): $0 - $35 (free tier: 1 pod, 1M vectors)
OpenAI (10K searches): ~$0.30 (if using LLM)
HuggingFace: Free (open models)
Infrastructure: $0 (local dev)
────────────────
Total: $0 - $50/month for learning
```

---

## Common Pitfalls & Solutions

### Problem: Poor search results
**Solutions:**
- [ ] Try a better embedding model (all-mpnet-base-v2)
- [ ] Adjust chunk size (try 256 or 1024)
- [ ] Add more overlap between chunks
- [ ] Check that query is being embedded correctly
- [ ] Verify documents are actually in Pinecone

### Problem: Slow ingestion
**Solutions:**
- [ ] Use batch embedding (already in code)
- [ ] Process documents in parallel
- [ ] Use a faster embedding model
- [ ] Skip re-embedding if already cached

### Problem: High costs
**Solutions:**
- [ ] Use Pinecone free tier (covers learning)
- [ ] Use open-source embedding models (free)
- [ ] Use local LLM instead of OpenAI
- [ ] Implement aggressive caching

### Problem: Hallucinations in LLM answers
**Solutions:**
- [ ] Provide more context to LLM
- [ ] Use better search results (fix retrieval)
- [ ] Add instruction to cite sources
- [ ] Use "zero-shot" style prompting

---

## Environment Setup Checklist

### Before You Start:
- [ ] Python 3.9+ installed
- [ ] pip and virtualenv available
- [ ] Pinecone account created (free tier)
- [ ] OpenAI account (if using GPT, optional for MVP)
- [ ] HuggingFace account (optional, for gated models)

### First Time Setup:
```powershell
# Navigate to project
cd c:\Users\bumpo\BookClub\PersonalKnowledgeAssistant

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Create .env file
cp backend\.env.example backend\.env
# Edit backend\.env with your API keys

# Install dependencies
pip install -r backend\requirements.txt

# Test setup
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

---

## Testing Strategy

### Unit Tests to Write:
1. **Embeddings**: 
   - Verify embedding dimension
   - Test batch processing
   - Verify cosine similarity works

2. **Chunking**:
   - Test overlap works correctly
   - Test metadata preservation
   - Test various document sizes

3. **Search**:
   - Test exact match retrieval
   - Test no results handling
   - Test filtering

### Integration Tests:
1. Full pipeline: ingest document → search → verify results
2. Multiple documents: ensure no cross-contamination
3. Large documents: test chunking and batch processing

### Manual Testing:
1. Upload sample PDF
2. Search with multiple queries
3. Verify metadata in results
4. Check Pinecone console for vectors

---

## Development Workflow

```
Day 1: Setup + Core modules (embeddings, pinecone)
Day 2: Document loading + chunking
Day 3: API endpoints
Day 4: Testing + CLI
Day 5: Frontend basics
Day 6+: Polish + advanced features
```

### Each Day:
1. Write code
2. Test locally (CLI first, then API)
3. Check Pinecone console for vectors
4. Verify search results manually
5. Commit to git

---

## Useful Commands

### Python Environment:
```powershell
# Activate
.\venv\Scripts\Activate.ps1

# Deactivate
deactivate

# Update requirements
pip freeze > backend\requirements.txt

# Install from requirements
pip install -r backend\requirements.txt
```

### Running Servers:
```powershell
# Start FastAPI
python backend\main.py

# In another terminal, test:
curl http://localhost:8000/api/health
```

### Testing:
```powershell
# Run CLI tool
python backend\cli\main.py ingest "path/to/file.txt"
python backend\cli\main.py search "your question here"
```

---

## Deployment Preview (Don't do yet!)

### Local Development:
- FastAPI backend on localhost:8000
- React frontend on localhost:3000
- Pinecone cloud (serverless)

### Simple Deployment:
- **Backend**: Railway.app, Render, or Heroku (free tier)
- **Frontend**: Vercel (free)
- **Database**: Pinecone serverless (managed)

### Enterprise Deployment:
- Docker containers
- Kubernetes orchestration
- Self-hosted vector DB
- Private LLM model

---

## Resources & Learning Links

### Official Documentation:
- [Pinecone Docs](https://docs.pinecone.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [LangChain Python](https://python.langchain.com/)

### Tutorials to Watch:
- "Building RAG with Python" series
- "Vector Databases Explained"
- "Embeddings Tutorial"
- "FastAPI Crash Course"

### Useful Articles:
- How RAG works
- Chunking strategies
- Prompt engineering tips
- Vector similarity search

---

## Next Step: Create Project Structure!

Ready to start? Follow the IMPLEMENTATION_GUIDE.md step by step.

Good luck! 🚀
