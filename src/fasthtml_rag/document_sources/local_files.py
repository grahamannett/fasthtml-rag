"""Local file system document source implementation."""

import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterator, Dict, List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from fasthtml_rag.document_sources.base import (
    Document,
    DocumentSource,
    DocumentSourcePlugin,
)


class LocalFileSource(DocumentSource):
    """Document source for local file system."""

    def __init__(self, root_dir: str, file_patterns: List[str] = None):
        self.root_dir = Path(root_dir)
        self.file_patterns = file_patterns or ["*.txt", "*.md", "*.pdf"]
        self._watch_callbacks = []

    @property
    def source_type(self) -> str:
        return "local_files"

    def _create_document(self, file_path: Path) -> Document:
        """Create a document instance from a file."""
        rel_path = file_path.relative_to(self.root_dir)
        stats = file_path.stat()

        return Document(
            id=str(rel_path),
            content="",  # Content loaded on demand
            metadata={
                "path": str(rel_path),
                "size": stats.st_size,
                "extension": file_path.suffix,
            },
            source_type=self.source_type,
            created_at=datetime.fromtimestamp(stats.st_ctime),
            updated_at=datetime.fromtimestamp(stats.st_mtime),
        )

    def list_documents(self) -> List[Document]:
        """List all documents in the root directory."""
        documents = []
        for pattern in self.file_patterns:
            for file_path in self.root_dir.rglob(pattern):
                if file_path.is_file():
                    documents.append(self._create_document(file_path))
        return documents

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by its ID (relative path)."""
        file_path = self.root_dir / doc_id
        if not file_path.is_file():
            return None

        doc = self._create_document(file_path)
        with open(file_path, "r") as f:
            doc.content = f.read()
        return doc

    def stream_document_chunks(
        self, doc: Document
    ) -> Iterator[tuple[str, Dict]]:
        """Stream document chunks with metadata."""
        file_path = self.root_dir / doc.metadata["path"]

        with open(file_path, "r") as f:
            content = f.read()

        # Simple chunking by paragraphs - can be made more sophisticated
        chunks = content.split("\n\n")
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue

            chunk_id = hashlib.md5(f"{doc.id}:{i}".encode()).hexdigest()
            metadata = {
                **doc.metadata,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            yield chunk, metadata

    def watch_for_changes(self) -> Iterator[List[Document]]:
        """Watch for file changes."""
        queue = []

        class Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory:
                    queue.append([Path(event.src_path)])

        observer = Observer()
        observer.schedule(Handler(), str(self.root_dir), recursive=True)
        observer.start()

        try:
            while True:
                if queue:
                    changed_paths = queue.pop(0)
                    yield [self._create_document(p) for p in changed_paths]
        finally:
            observer.stop()
            observer.join()


class LocalFilePlugin(DocumentSourcePlugin):
    """Plugin for local file source."""

    def __init__(self, root_dir: str = None, file_patterns: List[str] = None):
        self.root_dir = root_dir or os.getcwd()
        self.file_patterns = file_patterns

    def create_source(self) -> DocumentSource:
        return LocalFileSource(self.root_dir, self.file_patterns)
