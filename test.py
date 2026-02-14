from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import torch
import numpy as np
import json

# Model name
MODEL_NAME = "google/embeddinggemma-300m"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# The query to search for
query = "Where can I find a quiet place to read?"

documents = [
    "The cat slept on the warm windowsill.",
    "A bicycle leaned quietly against the old fence.",
    "Fresh coffee filled the kitchen with a rich aroma.",
    "The river moved slowly under the wooden bridge.",
    "A small robot rolled across the metal floor.",
    "The library was silent except for turning pages.",
    "Bright stars shimmered above the desert at night.",
    "A chef chopped vegetables with quick precision.",
    "The puppy chased a red ball across the yard.",
    "Rain tapped softly on the rooftop."
]

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token representation as embedding
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# Get embeddings for query and documents
query_embedding = get_embedding(query)
document_embeddings = np.stack([get_embedding(doc) for doc in documents])

# Compute cosine similarities
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(b, a)

similarities = cosine_similarity(query_embedding, document_embeddings)

# Sort and print results
print("\n" + "="*60)
print(f"Query: {query}")
print("="*60)

ranked_results = sorted(
    enumerate(documents),
    key=lambda x: similarities[x[0]],
    reverse=True
)

print("\nRanked Results (Most to Least Relevant):\n")
for rank, (idx, document) in enumerate(ranked_results, 1):
    score = similarities[idx]
    print(f"{rank}. Score: {score:.4f}")
    print(f"   {document}\n")

# Highlight the best answer
best_idx, best_document = ranked_results[0]
best_score = similarities[best_idx]
print("="*60)
print(f"Best Answer (Confidence: {best_score:.2%}):")
print(best_document)
print("="*60)

# Assign unique IDs and metadata for each document
vector_data = []
for i, (embedding, doc) in enumerate(zip(document_embeddings, documents)):
    vector_data.append({
        "id": f"doc-{i}",
        "values": embedding.tolist(),
        "metadata": {
            "text": doc,
            "source": "test.py",
            "index": i
        }
    })

# Save vectors to a JSON file for Pinecone upsert
with open("vectors.json", "w", encoding="utf-8") as f:
    json.dump(vector_data, f, ensure_ascii=False, indent=2)
