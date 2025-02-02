"""Base classes for document source plugins."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional
from datetime import datetime

@dataclass
class Document:
    """Represents a document with its metadata."""
    id: str
    content: str
    metadata: Dict
    source_type: str
    created_at: datetime
    updated_at: datetime
    
    @property
    def chunk_size(self) -> int:
        """Default chunk size for this document type."""
        return 1000

@dataclass
class SearchResult:
    """Represents a search result with relevance score."""
    document: Document
    score: float
    chunk_content: str
    chunk_metadata: Dict

class DocumentSource(ABC):
    """Abstract base class for document sources."""
    
    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the unique identifier for this source type."""
        pass
    
    @abstractmethod
    async def list_documents(self) -> List[Document]:
        """List all available documents from this source."""
        pass
    
    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a specific document by ID."""
        pass
    
    @abstractmethod
    async def stream_document_chunks(self, doc: Document) -> AsyncIterator[tuple[str, Dict]]:
        """Stream chunks of the document with their metadata."""
        pass
    
    @abstractmethod
    async def watch_for_changes(self) -> AsyncIterator[List[Document]]:
        """Watch for document changes and yield modified documents."""
        pass

class DocumentSourcePlugin:
    """Base class for document source plugins."""
    
    @classmethod
    def register(cls) -> None:
        """Register this plugin with the plugin registry."""
        from .plugins import plugin_registry
        plugin_registry.register_source(cls().create_source())
    
    @abstractmethod
    def create_source(self) -> DocumentSource:
        """Create and return a document source instance."""
        pass
