import os
from dotenv import load_dotenv

load_dotenv() # This looks for a file named .env in your root directory

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "knowledge-base")

# Model Configuration
EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"
LLM_MODEL_NAME = "gemma3:4b-it-qat"

# Local LLM URL (Assuming Ollama or similar local OpenAI-compatible server)
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/api/generate")

# Chunking Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50