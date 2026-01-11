# Personal Knowledge Assistant - Learning Curriculum

A structured guide to learning AI/ML concepts while building the app.

---

## Module 1: Understanding Embeddings (Days 1-2)

### Concepts to Learn

#### 1.1 What are Embeddings?
- Numerical representation of text
- Each word/sentence → vector of numbers
- Similar concepts = similar vectors
- Distance between vectors = similarity

**Exercise:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings
text1 = "The cat is sleeping"
text2 = "A cat is napping"
text3 = "The car is red"

emb1 = model.encode(text1)
emb2 = model.encode(text2)
emb3 = model.encode(text3)

print(f"Embedding 1 shape: {emb1.shape}")  # (384,)
print(f"Cosine similarity (text1, text2): {model.similarity(emb1, emb2)}")  # ~0.95
print(f"Cosine similarity (text1, text3): {model.similarity(emb1, emb3)}")  # ~0.20

# Key insight: Similar sentences have similar embeddings!
```

**Questions to explore:**
- Why 384 dimensions? (That's the model's output size)
- What's inside each number? (Model learned semantic features)
- How similar is "too similar"? (Depends on use case: 0.7 is good threshold)

#### 1.2 Cosine Similarity
- Measure of angle between vectors
- Range: -1 to 1 (0 to 1 for text embeddings)
- Ignores magnitude, only cares about direction
- Perfect for semantic similarity

**Exercise:**
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Two vectors
v1 = np.array([1, 0])
v2 = np.array([0.7, 0.7])
v3 = np.array([-1, 0])

# Similarity
print(cosine_similarity([v1], [v2])[0, 0])  # 0.7 (perpendicular)
print(cosine_similarity([v1], [v3])[0, 0])  # -1 (opposite)

# Key insight: Cosine ignores vector length!
np.array([1, 0]) and np.array([100, 0]) have similarity of 1.0
```

#### 1.3 Embedding Models Explained
- **Input**: Text (variable length)
- **Process**: Transformer neural network
- **Output**: Fixed-size vector (384, 768, 1536, etc.)

**Model Comparison:**
```
all-MiniLM-L6-v2
├─ Speed: Fast (< 0.01 sec per sentence)
├─ Quality: Good (0.86 MTEB score)
├─ Size: 22MB
└─ Best for: MVP, testing, rapid iteration

all-mpnet-base-v2
├─ Speed: Medium (0.05 sec per sentence)
├─ Quality: Better (0.88 MTEB score)
├─ Size: 109MB
└─ Best for: Production systems

bge-large-en-v1.5
├─ Speed: Slow (0.1 sec per sentence)
├─ Quality: Best (0.90+ MTEB score)
├─ Size: 335MB
└─ Best for: Domain-specific, high accuracy needed
```

**Exercise: Try different models**
```python
from sentence_transformers import SentenceTransformer
import time

models = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-mpnet-base-v2'
]

text = "Tell me about machine learning"

for model_name in models:
    model = SentenceTransformer(model_name)
    
    start = time.time()
    emb = model.encode(text)
    elapsed = time.time() - start
    
    print(f"{model_name}")
    print(f"  - Time: {elapsed:.3f}s")
    print(f"  - Dimensions: {len(emb)}")
    print(f"  - First 5 values: {emb[:5]}")
    print()
```

---

## Module 2: Vector Databases with Pinecone (Days 3-4)

### Concepts to Learn

#### 2.1 Why Vector Databases?
Traditional database: Keyword matching
```
SELECT * FROM documents WHERE content LIKE '%cat%'
→ Misses: "kitten", "feline", "furry animal"
```

Vector database: Semantic matching
```
SIMILARITY_SEARCH("cat", top_k=5)
→ Returns: "cat", "kitten", "feline", "animal", "pet"
→ All semantically related!
```

**Exercise: Understand the difference**
```python
# Keyword search - what does it miss?
documents = [
    "The cat is sleeping",
    "A kitten is playing",
    "Dogs are loyal animals",
    "The feline stretched"
]

query = "cat"

# Keyword approach (naive)
matches = [d for d in documents if "cat" in d.lower()]
print("Keyword matches:", matches)  # Only gets first + last

# Vector approach
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

query_emb = model.encode(query)
embeddings = [model.encode(d) for d in documents]

similarities = [
    model.similarity(query_emb, emb)[0][0].item() 
    for emb in embeddings
]

ranked = sorted(zip(documents, similarities), 
                key=lambda x: x[1], 
                reverse=True)
print("\nVector matches:")
for doc, score in ranked:
    print(f"  {score:.2f}: {doc}")
# Returns all 4 documents, ranked by relevance!
```

#### 2.2 How Pinecone Works

```
┌─────────────────────────────────────────┐
│         Upload your vectors             │
├─────────────────────────────────────────┤
│ Vector 1: [0.1, 0.2, 0.3] + metadata   │
│ Vector 2: [0.2, 0.1, 0.4] + metadata   │
│ Vector 3: [0.9, 0.1, 0.1] + metadata   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    Pinecone builds an index (ANN)       │
│  - Organizes vectors in N-dimensional   │
│  - Enables fast similarity search       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    Query with a vector                  │
│ "Find 5 vectors most similar to this"   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    Get results instantly                │
│ 1. [similarity: 0.95, metadata: {...}]  │
│ 2. [similarity: 0.92, metadata: {...}]  │
│ 3. [similarity: 0.88, metadata: {...}]  │
└─────────────────────────────────────────┘
```

**Key terms:**
- **Index**: The database of all your vectors
- **Upsert**: Add or update vectors
- **Query**: Search for similar vectors
- **ANN**: Approximate Nearest Neighbor (fast, not exact)
- **Metadata**: Extra info stored with each vector (source, text, timestamp)

#### 2.3 Pinecone Practical Setup

**Step 1: Create Account**
- Visit pinecone.io
- Sign up (free tier gives 1 pod, 1M vector limit)
- Create API key

**Step 2: Create Index**
```python
from pinecone import Pinecone

pc = Pinecone(api_key="your_key")

# Create once
pc.create_index(
    name="my-documents",
    dimension=384,  # Must match embedding model output
    metric="cosine",  # Options: cosine, euclidean, dotproduct
    spec=ServerlessSpec(cloud="aws", region="us-west-2")
)

index = pc.Index("my-documents")
```

**Step 3: Upsert Vectors**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Get embedding
text = "Machine learning is amazing"
embedding = model.encode(text).tolist()

# Upsert to Pinecone
index.upsert(vectors=[
    {
        "id": "doc-001",
        "values": embedding,
        "metadata": {
            "text": text,
            "source": "notes.txt",
            "created_at": "2025-01-10"
        }
    }
])
```

**Step 4: Query**
```python
# User asks a question
query = "Tell me about machine learning"
query_embedding = model.encode(query).tolist()

# Search Pinecone
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

# Process results
for match in results['matches']:
    print(f"Score: {match['score']:.2f}")  # 0-1 similarity
    print(f"Text: {match['metadata']['text']}")
    print(f"Source: {match['metadata']['source']}")
    print()
```

#### 2.4 Metadata Filtering

```python
# Store metadata with vectors
index.upsert(vectors=[
    {
        "id": "doc-001",
        "values": embedding,
        "metadata": {
            "text": "...",
            "source": "notes.txt",
            "tags": ["meeting", "q1", "strategy"]
        }
    }
])

# Query with filter
results = index.query(
    vector=query_embedding,
    top_k=5,
    filter={
        "tags": {"$in": ["meeting"]}  # Only search docs tagged "meeting"
    }
)
```

**Exercise: Build your first Pinecone index**
```python
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Initialize
pc = Pinecone(api_key="your_api_key")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create index
pc.create_index(
    name="learning-test",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-west-2")
)

index = pc.Index("learning-test")

# Add some documents
documents = [
    "Python is a programming language",
    "JavaScript runs in browsers",
    "Machine learning is AI",
    "Neural networks mimic brains"
]

vectors = []
for i, doc in enumerate(documents):
    emb = model.encode(doc).tolist()
    vectors.append({
        "id": f"doc-{i}",
        "values": emb,
        "metadata": {"text": doc}
    })

index.upsert(vectors=vectors)

# Test search
query = "What is programming?"
query_emb = model.encode(query).tolist()
results = index.query(vector=query_emb, top_k=3, include_metadata=True)

print("Search results:")
for r in results['matches']:
    print(f"  {r['score']:.2f}: {r['metadata']['text']}")
```

---

## Module 3: Document Processing & Chunking (Days 5-6)

### Concepts to Learn

#### 3.1 Why Chunking?

```
Long document (10,000 words)
    ↓
ONE embedding (loses lots of information)
    ↓
Poor search results ❌

Long document (10,000 words)
    ↓
Chunk into 20 pieces (512 words each)
    ↓
20 embeddings (each captures specific topic)
    ↓
Better search results ✓
```

#### 3.2 Chunking Strategies

**Strategy 1: Fixed size**
```python
def chunk_simple(text, chunk_size=512):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# Problem: Might cut sentences in half!
```

**Strategy 2: Sentence-aware (Better)**
```python
import re

def chunk_smart(text, chunk_size=512, overlap=50):
    # Split into sentences
    sentences = re.split(r'[.!?]\s+', text)
    
    chunks = []
    current_chunk = []
    word_count = 0
    
    for sentence in sentences:
        words = len(sentence.split())
        
        if word_count + words > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            word_count = words
        else:
            current_chunk.append(sentence)
            word_count += words
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Better: Preserves sentences
```

**Strategy 3: Overlapping chunks (Best)**
```python
def chunk_overlap(text, chunk_size=512, overlap=100):
    # Same as strategy 2, but with overlap
    # Last 100 words of chunk N = first 100 of chunk N+1
    # Benefits: Preserves context across chunks
    pass
```

**Exercise: Compare strategies**
```python
text = """Machine learning is a subset of artificial intelligence.
AI focuses on creating intelligent systems. Machine learning enables 
computers to learn from data. Neural networks are inspired by the brain.
Deep learning uses multiple layers of neural networks."""

# Try each strategy
chunks_1 = chunk_simple(text, 30)
chunks_2 = chunk_smart(text, 30)
chunks_3 = chunk_overlap(text, 30, overlap=10)

print("Fixed size:")
for i, c in enumerate(chunks_1):
    print(f"  [{i}] {c[:50]}...")

print("\nSentence-aware:")
for i, c in enumerate(chunks_2):
    print(f"  [{i}] {c[:50]}...")
```

#### 3.3 Processing Different File Types

**PDF Files**
```python
import pypdf

def load_pdf(filepath):
    text = ""
    with open(filepath, 'rb') as file:
        reader = pypdf.PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            text += f"[Page {page_num + 1}]\n"
            text += page.extract_text()
    return text

# Exercise: Try with a PDF from your computer
pdf_text = load_pdf("document.pdf")
print(f"Extracted {len(pdf_text)} characters")
```

**Web Content**
```python
import requests
from bs4 import BeautifulSoup

def load_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Remove scripts and styles
    for tag in soup(['script', 'style']):
        tag.decompose()
    
    text = soup.get_text()
    return '\n'.join(line.strip() for line in text.split('\n') if line.strip())

# Exercise: Try with a blog or Wikipedia
text = load_webpage("https://en.wikipedia.org/wiki/Machine_learning")
print(f"Extracted {len(text)} characters")
```

---

## Module 4: Building the Search Pipeline (Days 7-8)

### Concepts to Learn

#### 4.1 The Complete Flow

```
1. USER UPLOADS DOCUMENT
   ↓
2. LOAD DOCUMENT
   "Extract text from PDF/TXT/URL"
   ↓
3. CHUNK TEXT
   "Split into manageable pieces"
   ↓
4. EMBED CHUNKS
   "Convert each chunk to vector"
   ↓
5. STORE IN PINECONE
   "Save vectors + metadata"
   ↓
6. USER SEARCHES
   ↓
7. EMBED QUERY
   "Convert question to vector (SAME MODEL!)"
   ↓
8. SEARCH PINECONE
   "Find k most similar chunks"
   ↓
9. RETRIEVE RESULTS
   "Get full text + metadata"
   ↓
10. RETURN TO USER
```

**Critical insight**: Use the SAME embedding model for documents and queries!

#### 4.2 Building Step by Step

**Step 1: Create embedding wrapper**
```python
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_text(self, text):
        return self.model.encode(text).tolist()
    
    def embed_batch(self, texts):
        # Much faster for multiple texts!
        return [
            emb.tolist() 
            for emb in self.model.encode(texts)
        ]

# Test
embedder = Embedder()
emb = embedder.embed_text("Hello world")
print(f"Embedding length: {len(emb)}")  # 384
```

**Step 2: Create document processor**
```python
class DocumentProcessor:
    def __init__(self, embedder):
        self.embedder = embedder
    
    def process(self, filepath):
        # Load
        text = load_file(filepath)
        
        # Chunk
        chunks = chunk_smart(text, chunk_size=512)
        
        # Embed
        embeddings = self.embedder.embed_batch([c for c in chunks])
        
        # Prepare for storage
        vectors = [
            {
                "id": f"{filepath}_{i}",
                "values": emb,
                "metadata": {
                    "text": chunk,
                    "source": filepath,
                    "chunk_index": i
                }
            }
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        
        return vectors

# Test
processor = DocumentProcessor(embedder)
vectors = processor.process("notes.txt")
print(f"Created {len(vectors)} vectors")
```

**Step 3: Create search engine**
```python
class SearchEngine:
    def __init__(self, embedder, pinecone_index):
        self.embedder = embedder
        self.index = pinecone_index
    
    def search(self, query, top_k=5):
        # Embed query
        query_emb = self.embedder.embed_text(query)
        
        # Search
        results = self.index.query(
            vector=query_emb,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        formatted = []
        for match in results['matches']:
            formatted.append({
                'score': match['score'],
                'text': match['metadata']['text'],
                'source': match['metadata']['source']
            })
        
        return formatted

# Test
engine = SearchEngine(embedder, index)
results = engine.search("What is machine learning?")
for r in results:
    print(f"{r['score']:.2f}: {r['text'][:100]}")
```

---

## Module 5: API & Interface (Days 9-10)

### Concepts to Learn

#### 5.1 Building a FastAPI Backend

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search(request: SearchRequest):
    results = search_engine.search(
        request.query, 
        top_k=request.top_k
    )
    return {"results": results}

@app.post("/ingest")
async def ingest(file: UploadFile):
    # Save file, process, upsert to Pinecone
    return {"status": "success", "chunks": count}
```

#### 5.2 Creating a CLI Interface

```python
import typer

app = typer.Typer()

@app.command()
def ingest(filepath: str):
    vectors = processor.process(filepath)
    index.upsert(vectors=vectors)
    typer.echo(f"✓ Ingested {len(vectors)} chunks")

@app.command()
def search(query: str, top_k: int = 5):
    results = engine.search(query, top_k)
    for r in results:
        typer.echo(f"[{r['score']:.2f}] {r['source']}")
        typer.echo(f"  {r['text'][:100]}...")

if __name__ == "__main__":
    app()
```

Usage:
```bash
python cli.py ingest "document.txt"
python cli.py search "your question here"
```

---

## Module 6: LLM Integration (Days 11-12)

### Concepts to Learn

#### 6.1 Prompt Engineering for RAG

```python
def create_prompt(query, context):
    return f"""
You are a helpful assistant. Answer the following question based on 
the provided context. If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:
"""
```

#### 6.2 Calling OpenAI

```python
import openai

openai.api_key = "your_key"

def answer_question(query, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful knowledge assistant."
            },
            {
                "role": "user",
                "content": create_prompt(query, context)
            }
        ]
    )
    return response['choices'][0]['message']['content']
```

#### 6.3 Complete RAG Pipeline

```python
def answer_with_context(query):
    # 1. Search
    search_results = engine.search(query, top_k=5)
    
    # 2. Assemble context
    context = "\n\n".join([
        f"From {r['source']}:\n{r['text']}"
        for r in search_results
    ])
    
    # 3. Generate answer
    answer = answer_question(query, context)
    
    # 4. Return with sources
    return {
        "answer": answer,
        "sources": [r['source'] for r in search_results],
        "scores": [r['score'] for r in search_results]
    }
```

---

## Module 7: Testing & Evaluation (Days 13-14)

### Concepts to Learn

#### 7.1 Testing Your Embeddings

```python
def test_embedding_quality():
    # Test 1: Similar documents should have high similarity
    doc1 = "The cat sat on the mat"
    doc2 = "A cat was sitting on the mat"
    
    emb1 = embedder.embed_text(doc1)
    emb2 = embedder.embed_text(doc2)
    
    similarity = cosine_similarity([emb1], [emb2])[0, 0]
    assert similarity > 0.8, "Similar docs should have high similarity"
    
    # Test 2: Different documents should have low similarity
    doc3 = "The car is red"
    emb3 = embedder.embed_text(doc3)
    
    similarity2 = cosine_similarity([emb1], [emb3])[0, 0]
    assert similarity2 < 0.5, "Different docs should have low similarity"
```

#### 7.2 Evaluating Search Quality

```python
def evaluate_search():
    # Create test queries and expected results
    test_cases = [
        {
            "query": "machine learning",
            "expected_source": "ai_notes.txt",
            "should_rank": 1  # Should be in top 1
        }
    ]
    
    for test in test_cases:
        results = engine.search(test['query'], top_k=5)
        sources = [r['source'] for r in results]
        
        rank = next(
            (i for i, s in enumerate(sources) 
             if s == test['expected_source']),
            -1
        )
        
        assert rank >= 0 and rank < test['should_rank'], \
            f"Query '{test['query']}' failed"
```

#### 7.3 Metrics to Track

```
MRR (Mean Reciprocal Rank):
- If correct result is rank 1: MRR = 1.0
- If correct result is rank 5: MRR = 0.2
- Goal: MRR > 0.7

NDCG (Normalized Discounted Cumulative Gain):
- Weighs top results more heavily
- Accounts for partial relevance
- Goal: NDCG > 0.8

Latency:
- Ingestion: < 5 sec per document
- Search: < 500ms
- End-to-end: < 2 sec with LLM
```

---

## Learning Checkpoints

### After Module 1 ✓
- [ ] Understand what embeddings are
- [ ] Know why they're useful
- [ ] Can generate embeddings with code
- [ ] Understand cosine similarity

### After Module 2 ✓
- [ ] Have Pinecone account
- [ ] Created a Pinecone index
- [ ] Uploaded vectors
- [ ] Searched Pinecone

### After Module 3 ✓
- [ ] Can load PDF, TXT, and web content
- [ ] Understand chunking strategies
- [ ] Can chunk documents properly
- [ ] Know why metadata matters

### After Module 4 ✓
- [ ] Built complete ingestion pipeline
- [ ] Built search engine
- [ ] Can ingest document → search it
- [ ] Results are relevant

### After Module 5 ✓
- [ ] Created FastAPI server
- [ ] Have working API endpoints
- [ ] Built CLI tool
- [ ] Can use both interfaces

### After Module 6 ✓
- [ ] Integrated with LLM
- [ ] Understand RAG pipeline
- [ ] Can answer questions with context
- [ ] Sources are properly cited

### After Module 7 ✓
- [ ] Can evaluate quality
- [ ] Know how to improve results
- [ ] Can deploy with confidence
- [ ] Ready for production!

---

## Tips for Success

1. **Start Small**: Use tiny test documents first
2. **Test Each Component**: Don't build everything at once
3. **Monitor Pinecone**: Check console to verify vectors
4. **Try Different Models**: Compare embedding quality
5. **Read Error Messages**: They're usually helpful
6. **Ask Questions**: Community forums are friendly
7. **Keep Learning**: AI/ML moves fast!

---

## Next Steps

- Choose a start date
- Set aside ~1 hour/day
- Follow modules in order
- Don't skip the exercises
- Build something useful!

Good luck! 🚀
