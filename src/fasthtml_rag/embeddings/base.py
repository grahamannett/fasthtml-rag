"""Base classes for embedding models."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of texts into vectors."""
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query text."""
        pass
