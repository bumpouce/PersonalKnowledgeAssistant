# Personal Knowledge Assistant - Implementation Guide

## Quick Start (Getting Started Today)

### 1. Environment Setup
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Create requirements.txt with core dependencies
```

### 2. Key Concepts Explained

#### What are Vector Embeddings?
- Text → Numbers: "The cat sat" becomes [0.12, 0.45, -0.23, ...]
- Semantic meaning preserved: Similar words have similar numbers
- Enables: Similarity comparison, clustering, semantic search

#### Why Pinecone?
- Managed vector database (no server management)
- Fast similarity search (ANN - Approximate Nearest Neighbor)
- Metadata filtering (tag documents, filter results)
- Serverless scalability
- Free tier perfect for learning

#### Why HuggingFace Embeddings?
- Pre-trained models ready to use
- Open source (own your data)
- Multiple models for different trade-offs
- Easy to experiment and compare

---

## Step-by-Step Build Plan

### STEP 1: Initialize Project Structure

```
PersonalKnowledgeAssistant/
├── SYSTEM_DESIGN.md           # This design document
├── IMPLEMENTATION.md           # This file
├── backend/
│   ├── main.py                # FastAPI app entry point
│   ├── config.py              # Environment variables, settings
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example            # Template for environment vars
│   ├── core/
│   │   ├── embeddings.py       # HuggingFace embedding logic
│   │   ├── pinecone_client.py  # Pinecone database client
│   │   └── llm_client.py       # LLM integration (OpenAI)
│   ├── ingestion/
│   │   ├── document_loader.py  # PDF, text, web loaders
│   │   ├── text_splitter.py    # Chunking logic
│   │   └── pipeline.py         # End-to-end ingestion
│   ├── retrieval/
│   │   ├── search.py           # Semantic search logic
│   │   └── reranker.py         # Optional: improve results
│   ├── api/
│   │   ├── routes.py           # API endpoints
│   │   └── schemas.py          # Request/response models
│   └── cli/
│       └── main.py             # CLI interface for testing
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   └── pages/
│   ├── package.json
│   └── tailwind.config.js
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── README.md
```

---

### STEP 2: Install & Configure Dependencies

#### Create `backend/requirements.txt`:
```
# Core framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Vector embeddings
sentence-transformers==2.2.2
torch==2.1.1

# Vector database
pinecone-client==3.0.3

# LLM integration
openai==1.3.6
langchain==0.1.3
langchain-openai==0.0.5

# Document processing
pypdf==3.17.4
python-docx==0.8.11
requests==2.31.0
beautifulsoup4==4.12.2

# Utilities
python-dotenv==1.0.0
pydantic==2.5.0
pydantic-settings==2.1.0
```

#### Create `backend/.env.example`:
```
# Pinecone Configuration
PINECONE_API_KEY=your_api_key_here
PINECONE_ENVIRONMENT=us-west2-aws
PINECONE_INDEX_NAME=knowledge-assistant

# OpenAI Configuration
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Embedding Model (HuggingFace)
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# App Configuration
APP_DEBUG=false
MAX_CHUNK_SIZE=1024
CHUNK_OVERLAP=100
TOP_K_RESULTS=5
```

---

### STEP 3: Core Module - Embeddings (HuggingFace)

#### `backend/core/embeddings.py`:
```python
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed multiple texts efficiently."""
        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_tensor=False)
        return [emb.tolist() for emb in embeddings]
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim
```

**Learning Points:**
- `sentence-transformers` automatically downloads and caches models
- Batch processing is much faster than individual embedding
- Each model has fixed output dimension
- `convert_to_tensor=False` keeps in numpy for easier use

---

### STEP 4: Core Module - Pinecone Client

#### `backend/core/pinecone_client.py`:
```python
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Optional
import json
from datetime import datetime

class PineconeDB:
    def __init__(self, api_key: str, index_name: str, embedding_dim: int):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self.index = None
        self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist."""
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
        self.index = self.pc.Index(self.index_name)
    
    def upsert_vectors(self, vectors: List[tuple]):
        """
        Upsert vectors to Pinecone.
        vectors: List of (id, embedding, metadata) tuples
        """
        vectors_to_upsert = [
            (id, embedding, metadata) for id, embedding, metadata in vectors
        ]
        self.index.upsert(vectors=vectors_to_upsert)
    
    def search(self, embedding: List[float], top_k: int = 5, 
               filter: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar vectors.
        Returns: List of matches with scores and metadata
        """
        results = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        
        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            }
            for match in results.matches
        ]
    
    def delete(self, ids: List[str]):
        """Delete vectors by ID."""
        self.index.delete(ids=ids)
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        return self.index.describe_index_stats()
```

**Learning Points:**
- Pinecone indexes are created once, then managed
- "cosine" metric is standard for text (0-1 similarity)
- Metadata is stored with vectors for retrieval
- Filtering can reduce search space before vector search

---

### STEP 5: Document Ingestion - Text Splitter

#### `backend/ingestion/text_splitter.py`:
```python
from typing import List, Dict

class TextChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter (replace with nltk for production)
        text = text.replace("?", ". ").replace("!", ". ")
        sentences = text.split(". ")
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into overlapping chunks with metadata.
        
        Returns: List of chunks with text and metadata
        """
        # Clean text
        text = text.strip()
        
        # Split into sentences first
        sentences = self.split_by_sentences(text)
        
        # Create chunks from sentences
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": metadata or {}
                    })
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Don't forget last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "metadata": metadata or {}
            })
        
        return chunks
```

**Learning Points:**
- Chunking strategy significantly affects quality
- Overlap preserves context between chunks
- Metadata (source, page number) is crucial for attribution
- Sentence-aware chunking > naive splitting

---

### STEP 6: Document Ingestion - Loaders

#### `backend/ingestion/document_loader.py`:
```python
import pypdf
from pathlib import Path
from typing import List, Dict
import requests
from bs4 import BeautifulSoup

class DocumentLoader:
    @staticmethod
    def load_pdf(filepath: str) -> str:
        """Extract text from PDF."""
        text = ""
        with open(filepath, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text += f"[Page {page_num + 1}]\n"
                text += page.extract_text() + "\n"
        return text
    
    @staticmethod
    def load_text(filepath: str) -> str:
        """Load text file."""
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    
    @staticmethod
    def load_url(url: str) -> str:
        """Scrape text from webpage."""
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    @staticmethod
    def load_document(filepath_or_url: str) -> Dict[str, str]:
        """Auto-detect and load document."""
        if filepath_or_url.startswith('http'):
            content = DocumentLoader.load_url(filepath_or_url)
            source = filepath_or_url
        else:
            path = Path(filepath_or_url)
            if path.suffix.lower() == '.pdf':
                content = DocumentLoader.load_pdf(filepath_or_url)
            else:
                content = DocumentLoader.load_text(filepath_or_url)
            source = path.name
        
        return {"content": content, "source": source}
```

---

### STEP 7: Complete Ingestion Pipeline

#### `backend/ingestion/pipeline.py`:
```python
from typing import List, Dict
from datetime import datetime
from .document_loader import DocumentLoader
from .text_splitter import TextChunker
from ..core.embeddings import EmbeddingModel
from ..core.pinecone_client import PineconeDB

class IngestionPipeline:
    def __init__(self, embedding_model: EmbeddingModel, 
                 pinecone_db: PineconeDB, chunker: TextChunker):
        self.embedder = embedding_model
        self.db = pinecone_db
        self.chunker = chunker
    
    def ingest_document(self, filepath_or_url: str, tags: List[str] = None) -> Dict:
        """
        Complete pipeline: Load → Chunk → Embed → Store
        """
        # Load document
        doc = DocumentLoader.load_document(filepath_or_url)
        source = doc["source"]
        content = doc["content"]
        
        # Chunk text
        chunks = self.chunker.chunk_text(
            content, 
            metadata={
                "source": source,
                "created_at": datetime.now().isoformat(),
                "tags": tags or []
            }
        )
        
        # Embed chunks
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)
        
        # Prepare for Pinecone
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{source}_{i}".replace(".", "_").replace("/", "_")
            metadata = chunk["metadata"]
            metadata["text"] = chunk["text"]  # Store text in metadata
            vectors_to_upsert.append((vector_id, embedding, metadata))
        
        # Upsert to Pinecone
        self.db.upsert_vectors(vectors_to_upsert)
        
        return {
            "source": source,
            "chunks_created": len(chunks),
            "vectors_stored": len(vectors_to_upsert),
            "tags": tags
        }
```

---

### STEP 8: Query & Retrieval

#### `backend/retrieval/search.py`:
```python
from typing import List, Dict
from ..core.embeddings import EmbeddingModel
from ..core.pinecone_client import PineconeDB

class SemanticSearch:
    def __init__(self, embedding_model: EmbeddingModel, pinecone_db: PineconeDB):
        self.embedder = embedding_model
        self.db = pinecone_db
    
    def search(self, query: str, top_k: int = 5, 
               tags_filter: List[str] = None) -> List[Dict]:
        """
        Search knowledge base for relevant documents.
        """
        # Embed query using same model as documents
        query_embedding = self.embedder.embed_text(query)
        
        # Build filter if needed
        filter_dict = None
        if tags_filter:
            filter_dict = {"tags": {"$in": tags_filter}}
        
        # Search Pinecone
        results = self.db.search(
            embedding=query_embedding,
            top_k=top_k,
            filter=filter_dict
        )
        
        # Enrich results with formatted information
        search_results = []
        for result in results:
            search_results.append({
                "id": result["id"],
                "score": result["score"],  # 0-1 similarity
                "source": result["metadata"].get("source"),
                "text": result["metadata"].get("text"),
                "tags": result["metadata"].get("tags", []),
            })
        
        return search_results
```

---

### STEP 9: FastAPI Backend

#### `backend/main.py`:
```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from pathlib import Path
import tempfile

# Import our modules
from core.embeddings import EmbeddingModel
from core.pinecone_client import PineconeDB
from ingestion.text_splitter import TextChunker
from ingestion.pipeline import IngestionPipeline
from retrieval.search import SemanticSearch

# Global instances
embedding_model = None
pinecone_db = None
ingestion_pipeline = None
search_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global embedding_model, pinecone_db, ingestion_pipeline, search_engine
    
    print("Loading embedding model...")
    embedding_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    
    print("Connecting to Pinecone...")
    pinecone_db = PineconeDB(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name=os.getenv("PINECONE_INDEX_NAME", "knowledge-assistant"),
        embedding_dim=embedding_model.get_embedding_dimension()
    )
    
    chunker = TextChunker(chunk_size=512, overlap=100)
    ingestion_pipeline = IngestionPipeline(embedding_model, pinecone_db, chunker)
    search_engine = SemanticSearch(embedding_model, pinecone_db)
    
    print("✓ Ready!")
    yield
    
    # Shutdown
    print("Shutting down...")

app = FastAPI(title="Knowledge Assistant", lifespan=lifespan)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...), tags: str = ""):
    """Upload and ingest a document."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        # Ingest
        tags_list = [t.strip() for t in tags.split(",")] if tags else []
        result = ingestion_pipeline.ingest_document(tmp_path, tags=tags_list)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "message": f"Ingested {result['chunks_created']} chunks from {result['source']}",
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search(query: str, top_k: int = 5):
    """Search knowledge base."""
    try:
        results = search_engine.search(query, top_k=top_k)
        return {
            "status": "success",
            "query": query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    """Health check."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### STEP 10: CLI for Testing (Optional but useful for learning)

#### `backend/cli/main.py`:
```python
import typer
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from core.embeddings import EmbeddingModel
from core.pinecone_client import PineconeDB
from ingestion.text_splitter import TextChunker
from ingestion.pipeline import IngestionPipeline
from retrieval.search import SemanticSearch

app = typer.Typer()

# Initialize components
embedding_model = EmbeddingModel()
pinecone_db = PineconeDB(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name=os.getenv("PINECONE_INDEX_NAME", "knowledge-assistant"),
    embedding_dim=embedding_model.get_embedding_dimension()
)
chunker = TextChunker()
ingestion_pipeline = IngestionPipeline(embedding_model, pinecone_db, chunker)
search_engine = SemanticSearch(embedding_model, pinecone_db)

@app.command()
def ingest(filepath: str, tags: str = ""):
    """Ingest a document."""
    typer.echo(f"Ingesting {filepath}...")
    tags_list = [t.strip() for t in tags.split(",")] if tags else []
    result = ingestion_pipeline.ingest_document(filepath, tags=tags_list)
    typer.echo(f"✓ Success! Created {result['chunks_created']} chunks")

@app.command()
def search(query: str, top_k: int = 5):
    """Search knowledge base."""
    typer.echo(f"\nSearching for: {query}\n")
    results = search_engine.search(query, top_k=top_k)
    
    for i, result in enumerate(results, 1):
        typer.echo(f"[{i}] {result['source']} (score: {result['score']:.2f})")
        typer.echo(f"    {result['text'][:100]}...\n")

if __name__ == "__main__":
    app()
```

---

## Building Phase by Phase

### Phase 1 Checklist (MVP - 1 week):
- [ ] Set up Python project structure
- [ ] Install dependencies
- [ ] Create embedding model wrapper
- [ ] Set up Pinecone account and API key
- [ ] Create Pinecone client
- [ ] Implement text chunking
- [ ] Implement document loader for TXT files
- [ ] Create FastAPI with `/ingest` and `/search` endpoints
- [ ] Test with sample documents
- [ ] Create simple CLI tool

### Phase 2 Checklist (Better Ingestion):
- [ ] Add PDF support
- [ ] Add URL/web scraping
- [ ] Better metadata handling
- [ ] Implement tagging system
- [ ] Batch processing optimization

### Phase 3 Checklist (LLM Integration):
- [ ] Add OpenAI integration
- [ ] Create context assembly
- [ ] Implement `/answer` endpoint
- [ ] Add source attribution
- [ ] Test prompt engineering

### Phase 4 Checklist (Frontend):
- [ ] Create React app
- [ ] Build upload component
- [ ] Build search/chat interface
- [ ] Display results with sources
- [ ] Document management UI

### Phase 5+ Checklist (Advanced):
- [ ] User authentication
- [ ] Multiple collections
- [ ] Fine-tuned embeddings
- [ ] Re-ranking with cross-encoders
- [ ] Deployment

---

## Quick API Usage Examples

### Upload Document:
```bash
curl -X POST "http://localhost:8000/api/ingest" \
  -F "file=@document.txt" \
  -F "tags=project,important"
```

### Search:
```bash
curl -X POST "http://localhost:8000/api/search?query=What%20is%20this%20about&top_k=5"
```

---

## Learning Resources to Review During Build

1. **Vector Databases**: Search "semantic search tutorial"
2. **Embeddings**: HuggingFace model cards explain how models work
3. **LLM Prompting**: Learn about context windows and chain-of-thought
4. **RAG Pattern**: Research "Retrieval Augmented Generation"
5. **FastAPI**: Build incrementally, testing each endpoint

---

## Next: Start with Phase 1!
Create the project structure and start with `backend/core/embeddings.py`.
