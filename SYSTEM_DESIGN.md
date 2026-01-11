# Personal Knowledge Assistant - System Design

## Overview
A personal knowledge assistant that allows users to:
- Upload documents/notes (PDFs, text files, web content)
- Semantically search their knowledge base
- Get AI-powered answers based on their personal documents
- Learn about vector databases and embeddings in practice

---

## Architecture

### High-Level Flow
```
User Input
    ↓
Document Ingestion Layer
    ├─ PDF/Text Parser
    ├─ Text Chunking (semantic chunks)
    └─ Embedding Generation (HuggingFace)
    ↓
Vector Database (Pinecone)
    ├─ Store embeddings + metadata
    ├─ Enable similarity search
    └─ Organize by collection/topic
    ↓
Query Processing
    ├─ User question → embedding
    ├─ Semantic search in Pinecone
    └─ Retrieve relevant documents
    ↓
LLM Response Generation
    ├─ Combine context + question
    ├─ Generate personalized answer
    └─ Return with sources
    ↓
User (Web/CLI Interface)
```

---

## Key Components

### 1. **Document Ingestion Pipeline**
- **Input Handler**: Support multiple formats (PDF, TXT, DOCX, web URLs)
- **Text Extractor**: Extract and clean text from various formats
- **Text Chunker**: Split documents into semantic chunks (512-1024 tokens)
  - Strategy: Overlap chunks for better context
  - Preserve metadata (source, page number, timestamp)

### 2. **Embedding Generation**
- **Model Selection**: 
  - HuggingFace: `sentence-transformers/all-MiniLM-L6-v2` (fast, lightweight)
  - Alternative: `all-mpnet-base-v2` (higher quality, slower)
- **Batch Processing**: Embed multiple chunks efficiently
- **Caching**: Store embeddings locally to avoid re-embedding

### 3. **Vector Database - Pinecone**
- **Setup**:
  - Create index with dimension matching embedding model (~384 for MiniLM)
  - Enable metadata filtering
- **Data Structure**:
  ```json
  {
    "id": "doc_001_chunk_1",
    "values": [0.1, 0.2, ...], // embedding vector
    "metadata": {
      "source": "filename.pdf",
      "chunk_index": 1,
      "text": "actual text content",
      "created_at": "2025-01-10",
      "tags": ["topic1", "topic2"]
    }
  }
  ```
- **Operations**:
  - Upsert: Add/update embeddings
  - Query: Semantic similarity search (top-k)
  - Delete: Remove documents

### 4. **Query & Retrieval Engine**
- **Query Embedding**: Convert user question to embedding
- **Semantic Search**: Find most relevant chunks from Pinecone
- **Re-ranking**: Optional - re-score results for relevance
- **Context Assembly**: Combine top results with metadata

### 5. **LLM Integration**
- **Model Options**:
  - OpenAI GPT-4/3.5 (best quality)
  - HuggingFace hosted inference (self-hosted option)
  - Local models (Llama, Mistral)
- **Prompt Engineering**:
  - System prompt: Define assistant behavior
  - Context injection: Include retrieved documents
  - Chain-of-thought: Ask model to cite sources

### 6. **User Interface**
- **Frontend Options**:
  - Web app (React/Vue) - modern, accessible
  - CLI tool - simple, scriptable
  - Both for learning different interfaces
- **Features**:
  - Upload documents
  - Search/query interface
  - View sources and confidence scores
  - Manage collections
  - View chat history

---

## Technology Stack

### Backend
- **Language**: Python (rich ML/AI ecosystem)
- **Framework**: FastAPI (modern, async, easy to deploy)
- **LLM Integration**:
  - `langchain` or `llama-index` (abstracts LLM complexity)
  - Direct API calls to OpenAI/HuggingFace
- **Document Processing**:
  - `pypdf` - PDF extraction
  - `python-docx` - Word documents
  - `requests` - Web scraping
- **Embedding Model**:
  - `sentence-transformers` (HuggingFace)
- **Vector DB SDK**:
  - `pinecone-client` - Official Python SDK
- **Database**: SQLite/PostgreSQL for metadata backups

### Frontend
- **Web**: React + TypeScript, Tailwind CSS
- **CLI**: `rich` or `typer` for beautiful terminal UI
- **HTTP Client**: Axios or Fetch API

### DevOps/Deployment
- **Container**: Docker for consistency
- **Environment**: .env files for API keys
- **Deployment**: 
  - Local development
  - Cloud (Vercel for frontend, Railway/Heroku for backend)
  - Or single Docker container

---

## Data Flow Example: "Add a PDF"

```
1. User uploads "meeting-notes.pdf"
2. Extract text from PDF
3. Split into chunks:
   - Chunk 1: "Meeting on Q1 strategy..."
   - Chunk 2: "Action items include..."
4. Generate embeddings using HuggingFace
5. Prepare Pinecone records with metadata
6. Upsert to Pinecone index
7. Store metadata in local SQLite
8. Confirm to user: "3 chunks added from meeting-notes.pdf"
```

---

## Data Flow Example: "Answer a Question"

```
1. User asks: "What were the Q1 action items?"
2. Embed the question using same HuggingFace model
3. Query Pinecone with embedding vector
   - Pinecone returns: 5 most similar chunks with similarity scores
4. Retrieve full text and metadata from Pinecone results
5. Assemble prompt:
   - System: "You are a helpful knowledge assistant..."
   - Context: [Top 5 relevant chunks]
   - User: "What were the Q1 action items?"
6. Call LLM (OpenAI) with assembled prompt
7. Stream response back to user
8. Format response with:
   - Main answer
   - Sources cited (document + chunk)
   - Confidence/relevance scores
```

---

## Learning Objectives

### Vector Databases (Pinecone)
- ✓ Understand embeddings and vector similarity
- ✓ Index management and optimization
- ✓ Metadata filtering and tagging
- ✓ Scaling considerations
- ✓ Cost management (free tier learning)

### Embedding Models (HuggingFace)
- ✓ Download and run models locally
- ✓ Batch embedding for efficiency
- ✓ Model selection trade-offs (speed vs quality)
- ✓ Fine-tuning for domain-specific tasks
- ✓ Understanding tokenization

### RAG (Retrieval Augmented Generation)
- ✓ Semantic search vs keyword search
- ✓ Context assembly for LLMs
- ✓ Handling large documents
- ✓ Chain-of-thought prompting
- ✓ Source attribution

---

## Implementation Phases

### Phase 1: MVP (Week 1)
- [ ] Basic FastAPI backend with one endpoint
- [ ] Upload text file
- [ ] Embed using local HuggingFace model
- [ ] Store in Pinecone
- [ ] Simple semantic search
- [ ] CLI interface for testing

### Phase 2: Enhanced Ingestion (Week 2)
- [ ] PDF upload support
- [ ] Web URL scraping
- [ ] Better text chunking strategy
- [ ] Metadata enrichment

### Phase 3: Query & Response (Week 3)
- [ ] LLM integration (OpenAI)
- [ ] Prompt engineering
- [ ] Source attribution
- [ ] Streaming responses

### Phase 4: Web UI (Week 4)
- [ ] React frontend
- [ ] Upload interface
- [ ] Chat-like query interface
- [ ] Document management

### Phase 5: Polish & Learn (Week 5+)
- [ ] Error handling
- [ ] Rate limiting
- [ ] User authentication
- [ ] Analytics & monitoring
- [ ] Fine-tuning embeddings
- [ ] Advanced RAG techniques

---

## Key Considerations

### Performance
- Batch embeddings for faster ingestion
- Cache embeddings to avoid re-computing
- Pinecone's serverless pricing is efficient for small projects
- Consider chunking strategy for context quality

### Quality
- Experiment with different embedding models
- Use overlap in chunks to preserve context
- Re-rank retrieved results if needed
- Validate with domain-specific test queries

### Cost Management
- Pinecone: Free tier (1 project, 1 pod)
- OpenAI: $0.03/1K input tokens → estimate ~$10-20/month usage
- HuggingFace: Free open models
- Consider open-source LLM alternative later

### Security
- Secure API key management (.env files, environment variables)
- Input validation for uploads
- No sensitive data should be unencrypted
- Consider user auth if multi-user

---

## Useful Resources

### Documentation
- [Pinecone Docs](https://docs.pinecone.io/)
- [HuggingFace Models](https://huggingface.co/models)
- [LangChain Docs](https://python.langchain.com/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/)

### Libraries to Install
```bash
pip install pinecone-client
pip install sentence-transformers
pip install fastapi uvicorn
pip install python-multipart
pip install pypdf
pip install openai
pip install python-dotenv
pip install langchain
```

### Sample Projects
- LangChain's PrivateGPT
- Llama Index examples
- Haystack by Deepset

---

## Next Steps
1. Set up project structure and dependencies
2. Create .env configuration file
3. Build document ingestion pipeline
4. Connect to Pinecone
5. Create API endpoints
6. Build frontend
7. Deploy and learn!
