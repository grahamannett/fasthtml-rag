"""Ollama-based embedding model implementation."""

from typing import List
import numpy as np
import ollama

from .base import EmbeddingModel


class OllamaEmbedding(EmbeddingModel):
    """Embedding model using Ollama."""

    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name

    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts using Ollama."""
        embeddings = []
        for text in texts:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            embeddings.append(np.array(response["embedding"]))
        return embeddings

    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query text."""
        response = ollama.embeddings(model=self.model_name, prompt=query)
        return np.array(response["embedding"])
