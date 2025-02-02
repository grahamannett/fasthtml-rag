"""ChromaDB vector store implementation."""

from typing import List
import numpy as np
import chromadb
from chromadb.config import Settings

from .base import VectorStore
from ..document_sources.base import Document, SearchResult


class ChromaStore(VectorStore):
    """ChromaDB-based vector store implementation."""

    def __init__(self, persist_dir: str):
        self.client = chromadb.Client(
            Settings(persist_directory=persist_dir, anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection("documents")

    async def add_document(
        self, doc: Document, embeddings: List[np.ndarray], chunks: List[str]
    ) -> None:
        """Add document chunks to ChromaDB."""
        # Convert embeddings to list format
        embeddings_list = [e.tolist() for e in embeddings]

        # Create unique IDs for chunks
        chunk_ids = [f"{doc.id}_{i}" for i in range(len(chunks))]

        # Create metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "doc_id": doc.id,
                "chunk_index": i,
                "source_type": doc.source_type,
                **doc.metadata,
            }
            metadatas.append(metadata)

        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings_list,
            documents=chunks,
            ids=chunk_ids,
            metadatas=metadatas,
        )

    async def search(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> List[SearchResult]:
        """Search for similar chunks."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            doc = Document(
                id=metadata["doc_id"],
                content="",  # Content not needed for search results
                metadata={
                    k: v
                    for k, v in metadata.items()
                    if k not in ["doc_id", "chunk_index"]
                },
                source_type=metadata["source_type"],
                created_at=None,  # Not stored in ChromaDB
                updated_at=None,  # Not stored in ChromaDB
            )

            search_results.append(
                SearchResult(
                    document=doc,
                    score=1.0
                    - float(
                        results["distances"][0][i]
                    ),  # Convert distance to similarity score
                    chunk_content=results["documents"][0][i],
                    chunk_metadata=metadata,
                )
            )

        return search_results

    async def delete_document(self, doc_id: str) -> None:
        """Delete document chunks from ChromaDB."""
        # Find all chunks for this document
        results = self.collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    async def get_document_chunks(self, doc_id: str) -> List[tuple[str, dict]]:
        """Get all chunks for a document."""
        results = self.collection.get(where={"doc_id": doc_id})

        chunks = []
        for i, chunk_content in enumerate(results["documents"]):
            chunks.append((chunk_content, results["metadatas"][i]))

        return chunks
