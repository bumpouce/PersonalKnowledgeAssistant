import argparse
from Modules import config
from Modules.document import DocumentProcessor
from Modules.embedder import EmbedderClient
from Modules.vector_db import VectorDBClient
from Modules.llm import LLMClient

def main():
    parser = argparse.ArgumentParser(description="Local RAG Knowledge Manager CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: Upload
    upload_parser = subparsers.add_parser("upload", help="Upload a document to the knowledge base")
    upload_parser.add_argument("filepath", type=str, help="Relative path to the text file")

    # Command: Ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question against your knowledge base")
    ask_parser.add_argument("query", type=str, help="The question you want to ask")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize core modules
    embedder = EmbedderClient(config.EMBEDDING_MODEL_NAME)
    vector_db = VectorDBClient(config.PINECONE_API_KEY, config.PINECONE_INDEX_NAME)

    if args.command == "upload":
        doc_processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        
        print(f"Reading {args.filepath}...")
        text = doc_processor.read_file(args.filepath)
        if not text:
            return

        chunks = doc_processor.chunk_text(text)
        print(f"Created {len(chunks)} chunks. Embedding...")
        
        embeddings = embedder.embed_batch(chunks)
        vector_db.upload_chunks(chunks, embeddings, args.filepath)

    elif args.command == "ask":
        llm = LLMClient(config.LLM_MODEL_NAME, config.LOCAL_LLM_URL)
        
        print(f"Embedding query: '{args.query}'...")
        query_embedding = embedder.embed_text(args.query)
        
        print("Searching vector database...")
        top_chunks = vector_db.search(query_embedding, top_k=5)
        
        print("\n--- Top 5 Retrieved Chunks ---")
        for i, chunk in enumerate(top_chunks, 1):
            # Print a snippet of each chunk for readability
            print(f"{i}. {chunk[:100]}...") 
        print("------------------------------\n")
        
        print("Generating answer with local LLM...\n")
        answer = llm.ask_with_context(args.query, top_chunks)
        print(f"Answer:\n{answer}")

if __name__ == "__main__":
    main()