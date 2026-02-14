from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables from .env
load_dotenv()

# Model name
MODEL_NAME = "google/embeddinggemma-300m"

# Pinecone API key and index name
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "my-documents"
NAMESPACE = "example-namespace"

# Initialize Pinecone client and index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# The query to search for
query = "Where can I find a quiet place to read?"

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token representation as embedding
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# Get embedding for query
query_embedding = get_embedding(query)

# Query Pinecone for top 5 most similar vectors in the namespace
results = index.query(
    vector=query_embedding.tolist(),
    top_k=5,
    include_metadata=True,
    namespace=NAMESPACE
)

print("\n" + "="*60)
print(f"Query: {query}")
print("="*60)

print("\nTop 5 Pinecone Results (Most to Least Relevant):\n")
for rank, match in enumerate(results["matches"], 1):
    score = match["score"]
    text = match["metadata"].get("text", "")
    print(f"{rank}. Score: {score:.4f}")
    print(f"   {text}\n")

# Highlight the best answer
if results["matches"]:
    best = results["matches"][0]
    best_score = best["score"]
    best_text = best["metadata"].get("text", "")
    print("="*60)
    print(f"Best Answer (Confidence: {best_score:.2%}):")
    print(best_text)
    print("="*60)
