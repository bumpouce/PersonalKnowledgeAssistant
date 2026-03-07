from pinecone import Pinecone, ServerlessSpec
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Create once
pc.delete_index("my-new-documents")  # Clean up if it already exists
pc.create_index(
    name="my-new-documents",
    dimension=768,  # Must match embedding model output
    metric="cosine",  # Options: cosine, euclidean, dotproduct
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index("my-new-documents")

# Load vectors from file
with open("vectors.json", "r", encoding="utf-8") as f:
    vectors = json.load(f)

# Upsert vectors to Pinecone with a namespace
index.upsert(
    vectors=vectors,
    namespace="example-namespace"
)

print(f"Upserted {len(vectors)} vectors to Pinecone.")

