# src/embeddings.py
from pathlib import Path
from typing import List, Optional

import faiss  # type: ignore
import numpy as np
from numpy.typing import NDArray

from src.providers import get_openai_embeddings


class EmbeddingService:
    EMBEDDING_DIM = 1536  # OpenAI's ada-002 dimension

    def __init__(self):
        self.embeddings = get_openai_embeddings()
        self.index: Optional[faiss.Index] = None

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            numpy.ndarray: Embedding vector of shape (1536,)
        """
        if not text:
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        embedding = self.embeddings.embed_query(text)
        return np.array(embedding, dtype=np.float32)

    def embed_text_batch(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List[numpy.ndarray]: List of embedding vectors
        """
        if not texts:
            return []
        embeddings = self.embeddings.embed_documents(texts)
        return [np.array(emb, dtype=np.float32) for emb in embeddings]

    def load_faiss_index(self, index_path: str) -> faiss.Index:
        """Load a FAISS index from disk.

        Args:
            index_path: Path to the index file

        Returns:
            faiss.Index: Loaded FAISS index
        """
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        self.index = faiss.read_index(index_path)
        return self.index

    def save_faiss_index(self, index: faiss.Index, save_path: str) -> None:
        """Save FAISS index to disk.

        Args:
            index: FAISS index to save
            save_path: Path where to save the index
        """
        faiss.write_index(index, save_path)

    def create_index(self, dimension: int = EMBEDDING_DIM) -> faiss.Index:
        """Create a new FAISS index.

        Args:
            dimension: Dimension of vectors to index

        Returns:
            faiss.Index: New FAISS index
        """
        self.index = faiss.IndexFlatL2(dimension)
        return self.index
