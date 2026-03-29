import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class EmbedderClient:
    def __init__(self, model_name):
        print(f"Loading transformer model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Instruction required by EmbeddingGemma
        self.query_prefix = "task: search result | query: "
        self.doc_prefix = "task: search result | document: "

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_text(self, text: str, is_query: bool = True) -> list[float]:
        """Embeds a single string with the appropriate prefix."""
        prefix = self.query_prefix if is_query else self.doc_prefix
        full_text = prefix + text
        
        encoded_input = self.tokenizer(full_text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        # Perform pooling and normalization
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings[0].tolist()

    def embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """Embeds a list of strings."""
        prefix = self.query_prefix if is_query else self.doc_prefix
        full_texts = [prefix + t for t in texts]
        
        encoded_input = self.tokenizer(full_texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.tolist()