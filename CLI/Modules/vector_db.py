from pinecone import Pinecone, ServerlessSpec
import uuid
import time

class VectorDBClient:
    def __init__(self, api_key, index_name):
        """Initializes Pinecone client and ensures the index exists."""
        self.pc = Pinecone(api_key=api_key)
        
        # Check if the index exists; if not, create it
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"Index '{index_name}' not found. Creating it...")
            self.pc.create_index(
                name=index_name,
                dimension=768, # Gemma 300m dimensions
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'
                )
            )
            
            # Wait for index to be initialized
            print("Waiting for index to be ready...", end="", flush=True)
            while not self.pc.describe_index(index_name).status['ready']:
                print(".", end="", flush=True)
                time.sleep(2)
            print("\nIndex created successfully.")

        self.index = self.pc.Index(index_name)

    def upload_chunks(self, chunks: list[str], embeddings: list[list[float]], source_file: str):
        """Upserts vectors and their text metadata into Pinecone."""
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vector_id = str(uuid.uuid4())
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk, 
                    "source": source_file
                }
            })
        
        # Pinecone upsert
        self.index.upsert(vectors=vectors)
        print(f"Successfully uploaded {len(vectors)} chunks to Pinecone.")

    def search(self, query_embedding: list[float], top_k: int = 5):
        """Searches Pinecone and returns the top_k text chunks."""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract the text chunks from the metadata in the results
        matches = [match.metadata['text'] for match in results.matches]
        return matches