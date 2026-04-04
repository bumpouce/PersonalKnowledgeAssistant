from pinecone import Pinecone, ServerlessSpec
import hashlib
import time

class VectorDBClient:
    def __init__(self, api_key, index_name):
        """Initializes Pinecone client and ensures the index exists with correct dimensions."""
        self.pc = Pinecone(api_key=api_key)
        
        # Target dimension for google/embeddinggemma-300m is 768
        target_dimension = 768
        
        # Check if the index exists; if not, create it
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"Index '{index_name}' not found. Creating it with {target_dimension} dimensions...")
            self.pc.create_index(
                name=index_name,
                dimension=target_dimension,
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
        else:
            # Safety check: Verify existing index dimensions
            desc = self.pc.describe_index(index_name)
            if desc.dimension != target_dimension:
                print(f"CRITICAL: Index exists but has {desc.dimension} dimensions. Need {target_dimension}.")
                print("Please delete the index in the Pinecone dashboard and run this again.")

        self.index = self.pc.Index(index_name)

    def _generate_id(self, text: str) -> str:
        """Generates a deterministic ID based on the content of the text chunk."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def upload_chunks(self, chunks: list[str], embeddings: list[list[float]], source_file: str):
        """Upserts vectors and their text metadata into Pinecone using deterministic IDs."""
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            # Using a hash of the text prevents duplicate entries for the same content
            vector_id = self._generate_id(chunk)
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk, 
                    "source": source_file
                }
            })
        
        # Pinecone upsert: records with existing IDs are updated, not duplicated
        self.index.upsert(vectors=vectors)
        print(f"Successfully processed {len(vectors)} chunks in Pinecone (Deduplication enabled).")

    def search(self, query_embedding: list[float], top_k: int = 5):
        """Searches Pinecone and returns the top_k text chunks."""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        matches = [match.metadata['text'] for match in results.matches]
        return matches