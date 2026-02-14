# Troubleshooting Guide

Solutions for common problems when building the Personal Knowledge Assistant.

---

## Setup & Environment Issues

### Problem: "ModuleNotFoundError: No module named 'sentence_transformers'"

**Causes:**
- Dependencies not installed
- Wrong virtual environment activated
- pip install didn't complete

**Solutions:**
```powershell
# Make sure venv is activated
.\venv\Scripts\Activate.ps1

# Reinstall requirements
pip install --upgrade pip
pip install -r backend/requirements.txt

# Verify installation
python -c "from sentence_transformers import SentenceTransformer; print('✓ OK')"
```

---

### Problem: "No such file or directory: '.venv/lib/...'"

**Causes:**
- Using wrong shell (PowerShell vs CMD)
- Venv not activated properly

**Solutions:**
```powershell
# In PowerShell (Windows):
.\venv\Scripts\Activate.ps1

# Should see (venv) before prompt
# If not, try:
python -m venv venv  # Recreate venv
.\venv\Scripts\Activate.ps1
```

---

### Problem: Slow model download (first time)

**Causes:**
- Sentence Transformers downloads models on first use
- Large files (100-500 MB)

**Solutions:**
```python
# Pre-download models while your internet is good
from sentence_transformers import SentenceTransformer
import time

print("Downloading model...")
start = time.time()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
elapsed = time.time() - start
print(f"✓ Downloaded in {elapsed:.1f}s")
print(f"✓ Cached in: ~/.cache/huggingface/")
```

---

## Pinecone Issues

### Problem: "Invalid API key"

**Causes:**
- API key not set in .env
- Typo in API key
- Using someone else's key

**Solutions:**
```powershell
# Check .env file
cat backend\.env

# Make sure these lines exist:
# PINECONE_API_KEY=pk_...
# PINECONE_INDEX_NAME=knowledge-assistant

# Test connection
python -c "
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv('backend/.env')
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
print('✓ Connected!')
"
```

---

### Problem: "Index not found" or "Index does not exist"

**Causes:**
- Index name doesn't match configuration
- Index was deleted
- Using wrong Pinecone project

**Solutions:**
```python
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv('backend/.env')
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# List all your indexes
print("Your indexes:")
for idx in pc.list_indexes():
    print(f"  - {idx.name}")

# Create if missing
index_name = "knowledge-assistant"
if index_name not in [i.name for i in pc.list_indexes()]:
    print(f"Creating {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine"
    )
```

---

### Problem: "Dimension mismatch: expected 384, got 768"

**Causes:**
- Using different embedding model than index
- Index created with wrong dimensions

**Solutions:**
```python
from sentence_transformers import SentenceTransformer

# Check your model's dimension
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
# Should be 384

# If different, either:
# 1. Use same model everywhere
# 2. Recreate index with right dimension

# Recreate index:
from pinecone import Pinecone
pc = Pinecone(api_key=YOUR_KEY)
pc.delete_index("knowledge-assistant")
pc.create_index(name="knowledge-assistant", dimension=384, metric="cosine")
```

---

### Problem: "Quota exceeded" or rate limits

**Causes:**
- Free tier limits (1 pod, 1M vectors)
- Too many queries/writes simultaneously

**Solutions:**
```python
# Check index stats
index = pc.Index("knowledge-assistant")
stats = index.describe_index_stats()
print(f"Vectors in index: {stats.total_vector_count}")
print(f"Vector limit: 1,000,000 (free tier)")

# If at limit:
# Option 1: Delete old documents
index.delete(ids=old_vector_ids)

# Option 2: Switch to paid tier (very cheap: ~$1/month)
# Option 3: Create new index
```

---

### Problem: Search returns no results (empty list)

**Causes:**
- No vectors in index
- Filter too restrictive
- Query embedding mismatched

**Solutions:**
```python
# 1. Verify vectors are in index
stats = index.describe_index_stats()
print(f"Total vectors: {stats.total_vector_count}")

if stats.total_vector_count == 0:
    print("❌ No vectors! Need to ingest documents first")

# 2. Try without filters
results = index.query(
    vector=query_embedding,
    top_k=5,
    # filter=None  # Try without filter
)

# 3. Check similarity scores
if results['matches']:
    for m in results['matches']:
        print(f"Score: {m['score']:.2f}")  # Should be > 0.3
else:
    print("No matches found")

# 4. Verify same model used
# Document embedding model MUST == Query embedding model
```

---

## Embedding & Chunking Issues

### Problem: Search results are bad/irrelevant

**Causes:**
- Bad chunking strategy (chunks too large/small)
- Wrong embedding model
- Queries not matching document content
- Metadata loss

**Solutions:**
```python
# 1. Test the embedding model directly
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

text1 = "The cat sat on the mat"
text2 = "A kitten was sitting on the mat"
text3 = "The car is red"

emb1 = model.encode(text1)
emb2 = model.encode(text2)
emb3 = model.encode(text3)

sim_12 = model.similarity(emb1, emb2)[0][0]
sim_13 = model.similarity(emb1, emb3)[0][0]

print(f"Similar docs (1 vs 2): {sim_12:.2f}")  # Should be ~0.9
print(f"Different docs (1 vs 3): {sim_13:.2f}")  # Should be ~0.3

# If not, try different model:
model2 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# 2. Test chunking strategy
chunks = text_splitter.chunk_text(long_document)
print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n[Chunk {i}] ({len(chunk)} chars)")
    print(chunk[:100] + "...")

# 3. Verify metadata is preserved
for chunk in chunks:
    assert 'text' in chunk, "Missing text!"
    assert 'metadata' in chunk, "Missing metadata!"

# 4. Check actual data in Pinecone
results = index.query(vector=test_emb, top_k=3, include_metadata=True)
for r in results['matches']:
    print(f"\nID: {r['id']}")
    print(f"Score: {r['score']:.2f}")
    print(f"Text: {r['metadata'].get('text', 'MISSING')[:100]}")
```

---

### Problem: Chunks are too large/small

**Causes:**
- Wrong chunk_size parameter
- Overlap larger than chunk_size

**Solutions:**
```python
# Good chunk sizes
# 256 tokens: Too small, loses context
# 512 tokens: Good default for semantic search
# 1024 tokens: Good for LLM context
# 2048+ tokens: Risk of losing semantic meaning

from ingestion.text_splitter import TextChunker

# Test different sizes
for size in [256, 512, 1024]:
    chunker = TextChunker(chunk_size=size, overlap=50)
    chunks = chunker.chunk_text(sample_document)
    print(f"Size {size}: {len(chunks)} chunks")
    
    # Analyze first chunk
    first = chunks[0]['text']
    print(f"  First chunk: {len(first)} chars, {len(first.split())} words")
    print(f"  Preview: {first[:100]}...")
```

---

### Problem: PDF extraction is garbled

**Causes:**
- Scanned PDF (image-based, not text-based)
- Corrupted PDF
- Special encoding

**Solutions:**
```python
import pypdf

# Test PDF
try:
    with open("document.pdf", "rb") as f:
        reader = pypdf.PdfReader(f)
        
        # Check if PDF is text-based
        if reader.pages:
            text = reader.pages[0].extract_text()
            if not text or text.isspace():
                print("❌ PDF is image-based (scanned)")
                print("Solution: Use OCR library")
                # pip install pdf2image pytesseract
            else:
                print(f"✓ Text extracted: {len(text)} chars")
                print(f"Preview: {text[:200]}")
except Exception as e:
    print(f"❌ Error: {e}")

# For scanned PDFs, use OCR:
# from pdf2image import convert_from_path
# from pytesseract import image_to_string
```

---

## API & FastAPI Issues

### Problem: "Address already in use: ('0.0.0.0', 8000)"

**Causes:**
- Server already running on port 8000
- Previous server didn't shut down properly

**Solutions:**
```powershell
# Find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use different port
python backend/main.py  # Modify main.py:
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8001)  # Changed port
```

---

### Problem: "CORS policy: No 'Access-Control-Allow-Origin' header"

**Causes:**
- Frontend trying to call API from different domain
- CORS middleware not configured

**Solutions:**
```python
# In backend/main.py, ensure CORS is enabled:

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all (dev only!)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# For production, specify domains:
# allow_origins=["https://myapp.com", "https://www.myapp.com"]
```

---

### Problem: File upload not working

**Causes:**
- Multipart form data not configured
- File too large
- Wrong content-type header

**Solutions:**
```python
# Make sure you have this import
from fastapi import FastAPI, UploadFile, File, HTTPException

# Endpoint should look like:
@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        # Process...
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Test with curl:
# curl -X POST -F "file=@document.txt" http://localhost:8000/api/ingest
```

---

## LLM Integration Issues

### Problem: "Invalid API key" from OpenAI

**Causes:**
- API key not set in .env
- Wrong key format
- Key is expired/revoked

**Solutions:**
```powershell
# Check .env
cat backend\.env | grep OPENAI_API_KEY

# Should look like: sk-...

# Test connection
python -c "
import openai
import os
from dotenv import load_dotenv

load_dotenv('backend/.env')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Try to list models (doesn't cost money)
models = openai.Model.list()
print('✓ Connected to OpenAI')
"
```

---

### Problem: LLM responses are slow

**Causes:**
- Network latency
- Model is slow
- Large context

**Solutions:**
```python
# Use faster model
openai_model = "gpt-3.5-turbo"  # Fast
# vs
openai_model = "gpt-4"  # Slower but better

# Reduce context
results = search_engine.search(query, top_k=3)  # Instead of 10
context = "\n".join([r['text'] for r in results])

# Or use local model instead of OpenAI
# ollama run mistral  # Very fast, free
```

---

### Problem: LLM gives wrong/hallucinated answers

**Causes:**
- Irrelevant retrieved documents
- Poor prompt engineering
- Model doesn't have context

**Solutions:**
```python
# 1. Fix retrieval first!
results = engine.search(query, top_k=5)
for r in results:
    print(f"{r['score']:.2f}: {r['text'][:100]}")
    # Are these actually relevant?

# 2. Improve prompt
def create_prompt(query, context):
    return f"""You are a helpful knowledge assistant.
    
Answer ONLY using the provided context. If the answer is not in 
the context, clearly state that.

Context:
{context}

Question: {query}

Cite your sources. Format: (Source: document_name.pdf)

Answer:"""

# 3. Add instruction to cite sources
# 4. Reduce context size if too large
# 5. Use better retrieval model (different embedding model)
```

---

## General Debugging Tips

### Debug Strategy

1. **Isolate the problem**
   ```python
   # Test each component separately
   model = SentenceTransformer(...)  # Works?
   embedding = model.encode("test")  # Works?
   index.upsert(...)  # Works?
   index.query(...)  # Works?
   ```

2. **Add logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   logger = logging.getLogger(__name__)
   
   logger.debug(f"Embedding shape: {embedding.shape}")
   logger.debug(f"Search score: {result['score']}")
   ```

3. **Use print statements**
   ```python
   print(f"Query: {query}")
   print(f"Query embedding shape: {emb.shape}")
   print(f"Search results: {len(results)}")
   print(f"First result: {results[0] if results else 'None'}")
   ```

4. **Check Pinecone console**
   - Login to pinecone.io
   - View your index
   - See vector count
   - Verify data is stored

5. **Test in isolation**
   ```python
   # Don't test entire pipeline at once
   # Instead:
   # 1. Test embedding
   # 2. Test chunking
   # 3. Test Pinecone connection
   # 4. Test single document ingestion
   # 5. Test search on that document
   # 6. Test multiple documents
   # 7. Test LLM integration
   # ... only then test full pipeline
   ```

---

## Performance Issues

### Problem: Ingestion is slow

**Causes:**
- Embedding one chunk at a time
- Large documents
- Network latency to Pinecone

**Solutions:**
```python
# ❌ Slow: One at a time
for chunk in chunks:
    emb = model.encode(chunk)
    index.upsert([...])

# ✓ Fast: Batch embeddings
embeddings = model.encode(chunk_texts, batch_size=32)

# ✓ Faster: Batch Pinecone upsets
index.upsert(vectors=all_vectors_at_once)
```

---

### Problem: Search is slow

**Causes:**
- Large top_k value
- Filter is too broad
- Network latency

**Solutions:**
```python
# ✓ Faster
results = index.query(vector=emb, top_k=5)  # Not 100

# ✓ Much faster (if possible)
results = index.query(
    vector=emb,
    top_k=5,
    filter={"tags": {"$in": ["important"]}}
)
```

---

## Getting Help

### Before asking for help:

1. ✓ Check the error message
2. ✓ Search documentation
3. ✓ Test in isolation
4. ✓ Check logs for details
5. ✓ Try simpler test case

### Where to get help:

- **Pinecone**: https://community.pinecone.io/
- **HuggingFace**: https://github.com/UKPLab/sentence-transformers/
- **FastAPI**: https://discord.gg/VQjSZaeJmf
- **Python**: Stack Overflow

### When asking for help, include:

```
Problem: [Short description]

Environment:
- Python version
- OS (Windows/Mac/Linux)
- Relevant library versions

Error message:
[Full error traceback]

What I tried:
[Things you already attempted]

Code:
[Minimal code that reproduces the issue]
```

---

## Success Checklist

- [ ] Environment setup works
- [ ] Can import all libraries
- [ ] Pinecone connection works
- [ ] Can embed text
- [ ] Can upsert to Pinecone
- [ ] Can search Pinecone
- [ ] API endpoints work
- [ ] CLI tool works
- [ ] Search results are relevant
- [ ] LLM integration works

If you're stuck on any of these, refer to the relevant section above!

Good luck! 🚀
