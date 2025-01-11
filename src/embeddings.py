# src/embeddings.py
from typing import List

import faiss  # type: ignore
import numpy as np

from src.azure_providers import get_openai_embeddings


class EmbeddingService:
    def __init__(self):
        self.embeddings =  get_openai_embeddings()

    def embed_text(self, text: str) -> np.ndarray:
        """Returns embedding for a single doc."""
        embedding = self.embeddings.embed_query(text)
        return np.array(embedding)

    def embed_text_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embedding for a list of docs."""
        # Use the built-in batch method instead of list comprehension for better performance
        embeddings = self.embeddings.embed_documents(texts)
        return [np.array(emb) for emb in embeddings]

    def load_faiss_index(self, index_path: str):
        """Load a FAISS index from a file."""
        return faiss.read_index(index_path)


def test_embedding_service():
    """Test function to verify the embedding service works."""
    print("Testing EmbeddingService...")

    service = EmbeddingService()

    # Test single embedding
    test_text = "hello world"
    print(f"\nTesting single embedding with text: '{test_text}'")
    single_embedding = service.embed_text(test_text)
    print(f"Generated embedding with shape: {single_embedding.shape}")

    # Test batch embedding
    test_texts = ["hello world", "test text", "another example"]
    print(f"\nTesting batch embedding with {len(test_texts)} texts")
    batch_embeddings = service.embed_text_batch(test_texts)
    print(f"Generated {len(batch_embeddings)} embeddings, each with shape: {batch_embeddings[0].shape}")


if __name__ == "__main__":
    test_embedding_service()
