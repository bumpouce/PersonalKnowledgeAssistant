# Personal Knowledge Assistant 🧠

A learning project to build a personal knowledge management system using vector databases (Pinecone) and embedding models (HuggingFace). Perfect for understanding modern AI/ML concepts in practice.

## What This Project Teaches

✅ **Vector Embeddings** - How to convert text to meaningful numbers  
✅ **Vector Databases** - Semantic search without keywords  
✅ **LLM Integration** - Using AI models for answer generation  
✅ **RAG (Retrieval Augmented Generation)** - Combining search + LLMs  
✅ **Full Stack** - Backend, frontend, and deployment concepts  
✅ **Production Patterns** - Real-world system design  

## Quick Start

### 1. Read the Design Documents (10 min)
```
1. SYSTEM_DESIGN.md      → High-level architecture
2. QUICK_REFERENCE.md    → Visual diagrams & tech choices
3. IMPLEMENTATION_GUIDE  → Step-by-step code walkthrough
```

### 2. Setup Environment (5 min)
```powershell
# Navigate to project
cd c:\Users\bumpo\BookClub\PersonalKnowledgeAssistant

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r backend/requirements.txt
```

### 3. Configure API Keys (5 min)
```powershell
# Create .env file
cp backend\.env.example backend\.env

# Edit and add your keys:
# - PINECONE_API_KEY (free account at pinecone.io)
# - OPENAI_API_KEY (optional, for GPT integration)
```

### 4. Run MVP (This Week!)
Follow Phase 1 in IMPLEMENTATION_GUIDE.md to build:
- Document ingestion
- Semantic search
- Simple CLI interface

### 5. Expand (Next Weeks)
Phases 2-5 add:
- Web UI (React)
- LLM answer generation
- Advanced features

## Project Structure

```
PersonalKnowledgeAssistant/
├── SYSTEM_DESIGN.md           # Architecture & concepts
├── IMPLEMENTATION_GUIDE.md    # Step-by-step code
├── QUICK_REFERENCE.md         # Diagrams & reference
├── README.md                  # This file
│
├── backend/                   # Python FastAPI app
│   ├── main.py               # API server
│   ├── requirements.txt       # Dependencies
│   ├── .env.example          # Config template
│   ├── core/                 # Core logic
│   │   ├── embeddings.py     # HuggingFace integration
│   │   ├── pinecone_client.py # Vector DB client
│   │   └── llm_client.py     # LLM integration
│   ├── ingestion/            # Document processing
│   │   ├── document_loader.py
│   │   ├── text_splitter.py
│   │   └── pipeline.py
│   ├── retrieval/            # Search
│   │   └── search.py
│   ├── api/                  # FastAPI routes
│   └── cli/                  # CLI interface
│
├── frontend/                 # React app (optional)
│   ├── src/
│   ├── package.json
│   └── ...
│
└── docker/                   # Containerization
    ├── Dockerfile
    └── docker-compose.yml
```

## Key Concepts Explained

### What is a Vector Embedding?
Text → Numbers that preserve meaning
```
"The cat sat"      → [0.12, 0.45, -0.23, 0.87, ...]
"A cat was sitting" → [0.11, 0.44, -0.22, 0.86, ...]  ← Similar!
"The car stopped"   → [0.91, 0.02, 0.15, 0.33, ...]   ← Different
```

### Why Pinecone for Vector Databases?
- **Semantic Search**: Find similar documents without keywords
- **Fast**: Millions of vectors in milliseconds
- **Scalable**: Serverless, managed infrastructure
- **Flexible**: Metadata filtering, multiple indexes
- **Free Tier**: Perfect for learning!

### Why HuggingFace Embeddings?
- **Free**: Open-source models
- **Simple**: Pre-trained, ready to use
- **Local**: Your data stays with you
- **Flexible**: Switch models easily
- **Community**: Huge model library

### What is RAG?
**Retrieval Augmented Generation** = Search + LLM

1. User asks: "What was in that meeting?"
2. System **searches** documents for relevant context
3. System uses LLM to **generate** answer from context
4. User gets sources + personalized answer

## API Endpoints

### Upload Document
```bash
POST /api/ingest
- file: Document (PDF, TXT, etc.)
- tags: Comma-separated tags
```

### Search
```bash
POST /api/search
- query: Your question
- top_k: Number of results (default 5)
```

### Answer (with LLM)
```bash
POST /api/answer
- query: Your question
- top_k: Number of context chunks
```

## Learning Path

### Week 1: Foundations
- Understand embeddings and vector similarity
- Set up Pinecone and explore the console
- Ingest your first document
- Perform semantic search

### Week 2: Integration
- Connect to LLM (OpenAI)
- Generate answers from context
- Add source attribution
- Test with real documents

### Week 3: Interface
- Build simple web UI
- Upload and search UI
- Display sources
- Polish experience

### Week 4+: Advanced
- Fine-tune embeddings for your domain
- Implement re-ranking
- User authentication
- Deployment

## Example Usage

### Using CLI:
```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Ingest a document
python backend/cli/main.py ingest "notes.txt" --tags "meeting,q1"

# Search
python backend/cli/main.py search "What was discussed?"
```

### Using API:
```bash
# Start server
python backend/main.py

# In another terminal:
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this about?", "top_k": 5}'
```

## Common Questions

**Q: How much does this cost?**  
A: Free tier for learning! Pinecone free = 1 pod + 1M vectors. OpenAI = ~$0.30/1K tokens.

**Q: Can I use free/local LLMs?**  
A: Yes! Ollama, LLaMA 2, Mistral all work. Just slower than OpenAI.

**Q: How long to complete?**  
A: MVP in 1 week. Full app in 4-6 weeks of ~1 hour/day.

**Q: What if I get stuck?**  
A: 1) Check error message, 2) Search docs, 3) Test in CLI first, 4) Check Pinecone console.

**Q: Can I deploy this?**  
A: Yes! Vercel (frontend) + Railway/Render (backend) + Pinecone (DB) = ~$0/month free tier.

## Tools & Resources

### Essential Accounts (Free):
- [Pinecone](https://pinecone.io) - Vector database
- [OpenAI](https://openai.com) - LLM (optional, can use free models)
- [GitHub](https://github.com) - Version control

### Documentation:
- [Pinecone Docs](https://docs.pinecone.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://python.langchain.com/)

### Tutorials:
- "RAG with LangChain" by Harrison Chase
- "Vector Databases" by James Briggs
- "Building with AI" courses

## Next Steps

1. ✅ You're reading this - good!
2. 📖 Read `SYSTEM_DESIGN.md` 
3. 📖 Read `QUICK_REFERENCE.md`
4. 🔧 Follow `IMPLEMENTATION_GUIDE.md` Phase 1
5. 💻 Build and learn!

## Project Status

- [x] System Design
- [x] Implementation Guide
- [x] Code Examples
- [ ] MVP Implementation (You start here!)
- [ ] Testing Suite
- [ ] React Frontend
- [ ] Deployment

## Support

- Check the README and docs first
- Search existing issues
- Read error messages carefully
- Test with simple examples first
- Check Pinecone console to verify data

## Python Version Requirement
This project requires Python 3.10, 3.11, or 3.12. It is not compatible with Python 3.13 due to upstream library limitations.

---

**Built with** 🤍 for learning about modern AI/ML infrastructure.

**Last Updated**: January 10, 2026

**Status**: Ready to build! Start with Phase 1 of IMPLEMENTATION_GUIDE.md
