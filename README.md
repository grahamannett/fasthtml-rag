# FastHTML RAG Demo

A demonstration of building a RAG (Retrieval-Augmented Generation) application using FastHTML and Chroma Vector Store. This application showcases how to create an interactive chat interface that can reference and search through your documents for more accurate and context-aware responses.

![screenshot](https://github.com/grahamannett/fasthtml-rag/blob/main/docs/app.png?raw=true)

## Features

- 🚀 Built with FastHTML for rapid, modern web UI development
- 📝 Document management with support for various file types (txt, md, pdf)
- 🔍 RAG implementation using Chroma as the vector store backend
- 🤖 Ollama integration for embeddings and text generation
- 🔄 Real-time chat interface with streaming responses
- 📁 Local file system document source with file change watching

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai) installed and running locally
- [chromadb](https://github.com/chroma-core/chroma) for vector storage

## Installation

```bash
uv sync
```

## Usage

1. Start the application:

```bash
uv run src/fasthtml_rag/app.py
```

2. Open your browser and navigate to `http://localhost:3000`

3. Upload documents using the file upload interface

4. Start chatting! The application will use the uploaded documents as context for generating responses

## Project Structure

- `src/fasthtml_rag/`
  - `app.py`: Main application setup and FastHTML routes
  - `document_sources/`: Document source plugins and base classes
  - `embeddings/`: Embedding model implementations

## How it Works

1. **Document Processing**: When you upload documents, they are processed and chunked into smaller segments
2. **Embedding Generation**: Each chunk is converted into embeddings
3. **Vector Storage**: Embeddings are stored in a Chroma vector store for efficient similarity search
4. **RAG Process**: When you ask a question, the system:
   - Converts your query into an embedding
   - Finds relevant document chunks using similarity search
   - Uses the found context to generate an informed response
