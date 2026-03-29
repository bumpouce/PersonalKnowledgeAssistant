import requests
import json

class LLMClient:
    def __init__(self, model_name, api_url):
        self.model_name = model_name
        self.api_url = api_url

    def ask_with_context(self, question: str, context_chunks: list[str]):
        """Constructs a prompt with context and streams/returns the LLM response."""
        context_str = "\n---\n".join(context_chunks)
        
        prompt = f"""You are a helpful knowledge assistant. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.

Context:
{context_str}

Question: {question}
Answer:"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False 
        }

        response = requests.post(self.api_url, json=payload)
        response.raise_for_status()
        
        return response.json().get('response', '')