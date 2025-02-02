"""Base classes for vector stores."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from ..document_sources.base import Document, SearchResult

class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def add_document(self, doc: Document, embeddings: List[np.ndarray], chunks: List[str]) -> None:
        """Add a document's chunks and their embeddings to the store."""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        """Search for similar chunks using the query embedding."""
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        """Delete a document and its chunks from the store."""
        pass
    
    @abstractmethod
    async def get_document_chunks(self, doc_id: str) -> List[tuple[str, dict]]:
        """Get all chunks for a document."""
        pass
