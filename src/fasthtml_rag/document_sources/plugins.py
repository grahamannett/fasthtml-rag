"""Plugin registry for document sources."""

from typing import Dict, List, Type
from .base import DocumentSource


class PluginRegistry:
    """Registry for document source plugins."""

    def __init__(self):
        self._sources: Dict[str, DocumentSource] = {}

    def register_source(self, source: DocumentSource) -> None:
        """Register a document source."""
        # If source is already registered, just update it
        self._sources[source.source_type] = source

    def get_source(self, source_type: str) -> DocumentSource:
        """Get a document source by type."""
        if source_type not in self._sources:
            raise ValueError(f"Source type {source_type} not found")
        return self._sources[source_type]

    def list_sources(self) -> List[str]:
        """List all registered source types."""
        return list(self._sources.keys())


# Global plugin registry
plugin_registry = PluginRegistry()
