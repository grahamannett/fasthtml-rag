"""Example refactored for improved readability and type safety with Ollama integration and RAG support"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import ollama
from fasthtml.common import (
    H1,
    H2,
    Body,
    Button,
    Div,
    FastHTML,
    Form,
    Group,
    Input,
    Link,
    Script,
    picolink,
    serve,
    Table,
    Tr,
    Td,
    Th,
    P,
    Pre,
    UploadFile,
    Style,
)
from ollama import chat
from starlette.responses import StreamingResponse

from fasthtml_rag.document_sources.base import Document, SearchResult
from fasthtml_rag.document_sources.plugins import plugin_registry
from fasthtml_rag.document_sources.local_files import LocalFilePlugin
from fasthtml_rag.vectorstore.chroma import ChromaStore
from fasthtml_rag.embeddings.ollama import OllamaEmbedding

# Set up the app, including daisyui and tailwind
tlink = (Script(src="https://cdn.tailwindcss.com"),)
dlink = Link(
    rel="stylesheet",
    href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css",
)
sselink = Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")

# Custom styles for cyberpunk theme
custom_styles = Style("""
    @keyframes glow {
        0% { box-shadow: 0 0 5px #0ff, 0 0 10px #0ff, 0 0 15px #0ff; }
        100% { box-shadow: 0 0 10px #0ff, 0 0 20px #0ff, 0 0 30px #0ff; }
    }
    @keyframes scanline {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(100%); }
    }
    .cyberpunk-card {
        background: rgba(13, 17, 23, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 255, 0.1);
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.1);
    }
    .cyberpunk-card:hover {
        border-color: rgba(0, 255, 255, 0.3);
    }
    .neon-border {
        position: relative;
        overflow: hidden;
    }
    .neon-border::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #0ff, transparent);
        animation: scanline 2s linear infinite;
    }
    .chat-bubble-primary {
        background: rgba(0, 255, 255, 0.1) !important;
        border: 1px solid rgba(0, 255, 255, 0.2) !important;
    }
    .chat-bubble-secondary {
        background: rgba(255, 0, 255, 0.1) !important;
        border: 1px solid rgba(255, 0, 255, 0.2) !important;
    }
    .cyber-input {
        background: rgba(13, 17, 23, 0.8) !important;
        border: 1px solid rgba(0, 255, 255, 0.2) !important;
        color: #fff !important;
    }
    .cyber-input:focus {
        border-color: #0ff !important;
        animation: glow 2s ease-in-out infinite alternate;
    }
    .cyber-button {
        background: linear-gradient(45deg, #0ff, #00f) !important;
        border: none !important;
        color: #fff !important;
        text-shadow: 0 0 5px rgba(0, 255, 255, 0.5);
    }
    .cyber-button:hover {
        background: linear-gradient(45deg, #00f, #0ff) !important;
        animation: glow 1s ease-in-out infinite alternate;
    }
    .cyber-table {
        background: rgba(13, 17, 23, 0.6);
    }
    .cyber-table th {
        background: rgba(0, 255, 255, 0.1) !important;
        color: #0ff !important;
    }
    .cyber-table tr:hover {
        background: rgba(0, 255, 255, 0.05) !important;
    }
""")

app = FastHTML(hdrs=(tlink, dlink, sselink, custom_styles), live=True)

# Define constants and initialize components
MODEL_ID = "llama3.2:3b"
PERSIST_DIR = Path("data/chroma")
DOCS_DIR = Path("data/documents")

system_prompt = "You are a helpful and concise assistant. Use the provided context to answer questions."
messages = []  # List of message dicts

# Ensure directories exist
PERSIST_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize components
embedding_model = OllamaEmbedding(MODEL_ID)
vector_store = ChromaStore(str(PERSIST_DIR))

# Register document sources
LocalFilePlugin(str(DOCS_DIR)).register()


async def process_document(doc: Document) -> None:
    """Process a document and add it to the vector store."""
    chunks = []
    chunk_texts = []
    source = plugin_registry.get_source(doc.source_type)

    # Handle synchronous document chunks
    for chunk, metadata in source.stream_document_chunks(doc):
        chunks.append((chunk, metadata))
        chunk_texts.append(chunk)

    embeddings = await embedding_model.embed_texts(chunk_texts)
    await vector_store.add_document(doc, embeddings, chunk_texts)


def list_documents() -> List[Document]:
    """List all available documents."""
    source = plugin_registry.get_source("local_files")
    return source.list_documents()


async def get_relevant_context(query: str, top_k: int = 3) -> List[SearchResult]:
    """Get relevant context for a query."""
    query_embedding = await embedding_model.embed_query(query)
    return await vector_store.search(query_embedding, top_k=top_k)


async def message_generator() -> AsyncGenerator[str, None]:
    """Yields streaming chat responses from the Ollama model."""
    conversation = [{"role": "system", "content": system_prompt}]
    conversation.extend(messages)

    # Get relevant context from documents
    context_results = await get_relevant_context(messages[-1]["content"])
    if context_results:
        context = "\n\n".join(
            f"Content: {result.chunk_content}\nSource: {result.document.metadata['path']}"
            for result in context_results
        )
        system_context = f"Use this context to help answer the question:\n\n{context}"
        conversation.insert(1, {"role": "system", "content": system_context})

    # Stream the response
    response = chat.chat(
        model=MODEL_ID,
        messages=conversation,
        stream=True,
    )

    full_response = ""
    for chunk in response:
        if "message" in chunk and chunk["message"].get("content"):
            content = chunk["message"]["content"]
            full_response += content
            # Yield both the content and the message index for the client
            yield f"data: {{'content': '{content}', 'msg_idx': {len(messages)}}}\n\n"

    # Add the complete message to our history
    messages.append({"role": "assistant", "content": full_response})


@app.route("/send", methods=["POST"])
async def send(msg: str = "") -> StreamingResponse:
    """Handle sending a message."""
    if not msg:
        return "Message cannot be empty"

    # Add user message
    messages.append({"role": "user", "content": msg})
    user_msg_idx = len(messages) - 1

    # Add placeholder for assistant message
    messages.append({"role": "assistant", "content": ""})
    assistant_msg_idx = len(messages) - 1

    async def send_events():
        # Send the user message first
        yield f"data: {{'type': 'user', 'html': '{chat_message(user_msg_idx)}'}}\n\n"
        # Send initial assistant message
        yield f"data: {{'type': 'assistant', 'html': '{chat_message(assistant_msg_idx)}'}}\n\n"
        # Stream the response
        async for chunk in message_generator():
            yield chunk

    return StreamingResponse(
        send_events(),
        media_type="text/event-stream",
    )


def chat_message(msg_idx: int, **kwargs) -> Div:
    """Returns a Div element representing a chat message."""
    msg = messages[msg_idx]
    bubble_class = (
        "chat-bubble-primary" if msg["role"] == "user" else "chat-bubble-secondary"
    )
    chat_class = "chat-end" if msg["role"] == "user" else "chat-start"

    return Div(
        Div(
            msg["role"],
            cls="chat-header opacity-50 text-cyan-400"
            if msg["role"] == "user"
            else "chat-header opacity-50 text-fuchsia-400",
        ),
        Div(
            msg["content"],
            id=f"chat-content-{msg_idx}",
            cls=f"chat-bubble {bubble_class} prose max-w-full backdrop-blur-sm",
            **kwargs,
        ),
        id=f"chat-message-{msg_idx}",
        cls=f"chat {chat_class}",
    )


def chat_input() -> Input:
    """Returns the input component for user messages."""
    return Input(
        type="text",
        name="msg",
        id="msg-input",
        placeholder="Type a message",
        cls="input cyber-input w-full",
        hx_swap_oob="true",
    )


def upload_form() -> Form:
    """Returns the upload form component."""
    return Form(
        Group(
            Input(
                type="file",
                name="file",
                cls="file-input cyber-input w-full",
                accept=".txt,.md,.pdf",
            ),
            Button(
                "Upload",
                type="submit",
                cls="btn cyber-button",
            ),
            cls="join",
        ),
        cls="mb-4",
        hx_post="/upload",
        hx_target="#documents",
        hx_encoding="multipart/form-data",
    )


def document_table(documents: List[Document]) -> Div:
    """Returns the document table component."""
    if not documents:
        return Div(
            P(
                "No documents uploaded yet.",
                cls="text-center text-cyan-400 opacity-50",
            ),
            cls="cyberpunk-card rounded-lg p-4",
        )

    return Table(
        Tr(
            Th("Document"),
            Th("Size"),
            Th("Updated"),
            Th("Actions"),
            cls="bg-base-200",
        ),
        *[
            Tr(
                Td(doc.metadata["path"], cls="text-cyan-400"),
                Td(f"{doc.metadata['size'] / 1024:.1f} KB", cls="text-fuchsia-400"),
                Td(
                    doc.updated_at.strftime("%Y-%m-%d %H:%M"),
                    cls="text-cyan-400 opacity-75",
                ),
                Td(
                    Button(
                        "Delete",
                        cls="btn btn-error btn-xs cyber-button",
                        hx_delete=f"/documents/{doc.id}",
                        hx_target="#documents",
                    ),
                ),
            )
            for doc in sorted(documents, key=lambda d: d.updated_at, reverse=True)
        ],
        cls="table w-full cyber-table",
    )


@app.route("/")
async def index():
    """Render the main page."""
    documents = list_documents()

    return Body(
        # Background with a subtle cyberpunk pattern
        Style("""
            body {
                background-color: #0d1117;
                background-image: 
                    linear-gradient(45deg, rgba(0, 255, 255, 0.05) 25%, transparent 25%),
                    linear-gradient(-45deg, rgba(0, 255, 255, 0.05) 25%, transparent 25%),
                    linear-gradient(45deg, transparent 75%, rgba(0, 255, 255, 0.05) 75%),
                    linear-gradient(-45deg, transparent 75%, rgba(0, 255, 255, 0.05) 75%);
                background-size: 20px 20px;
                background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
            }
        """),
        H1(
            "🐌",
            cls="text-4xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-fuchsia-400",
        ),
        Div(
            Div(
                H2(
                    "Documents",
                    cls="text-xl font-semibold mb-4 text-cyan-400",
                ),
                upload_form(),
                Div(
                    document_table(documents),
                    id="documents",
                    cls="overflow-x-auto",
                ),
                cls="w-full lg:w-1/3 p-4 cyberpunk-card rounded-lg neon-border",
            ),
            Div(
                H2(
                    "Chat",
                    cls="text-xl font-semibold mb-4 text-fuchsia-400",
                ),
                Div(
                    *(chat_message(i) for i in range(len(messages))),
                    id="chat-messages",
                    cls="space-y-4 mb-4 h-[60vh] overflow-y-auto p-4 cyberpunk-card rounded-lg backdrop-blur-sm",
                ),
                Form(
                    chat_input(),
                    Button(
                        "Send",
                        type="submit",
                        cls="btn cyber-button ml-2",
                    ),
                    cls="flex space-x-2",
                    hx_post="/send",
                    hx_swap="beforeend",
                    hx_target="#chat-messages",
                    hx_ext="sse",
                    _="on submit set #msg-input.value to ''",
                ),
                cls="w-full lg:w-2/3 p-4 cyberpunk-card rounded-lg neon-border",
            ),
            cls="flex flex-col lg:flex-row gap-6",
        ),
        Script("""
            htmx.on("htmx:sseMessage", (e) => {
                const data = JSON.parse(e.detail.data);
                
                if (data.type === 'user' || data.type === 'assistant') {
                    // Handle new message HTML
                    const chatMessages = document.getElementById('chat-messages');
                    chatMessages.innerHTML += data.html;
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                } else if (data.content) {
                    // Handle streaming content
                    const msgElement = document.getElementById(`chat-content-${data.msg_idx}`);
                    if (msgElement) {
                        msgElement.textContent = msgElement.textContent + data.content;
                        msgElement.parentElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    }
                }
            });
        """),
        cls="container mx-auto p-6 max-w-7xl",
    )


@app.route("/upload", methods=["POST"])
async def handle_upload(file: UploadFile):
    """Handle file uploads."""
    if not file.filename:
        return "No file selected"

    # Save file to documents directory
    file_path = DOCS_DIR / file.filename
    file_content = await file.read()
    file_path.write_bytes(file_content)

    # Process the document
    source = plugin_registry.get_source("local_files")
    doc = source.get_document(file.filename)
    if doc:
        await process_document(doc)

    # Return updated document table
    documents = list_documents()
    return document_table(documents)


@app.route("/documents/{doc_id}", methods=["DELETE"])
async def delete_document(doc_id: str):
    """Delete a document."""
    # Delete from vector store
    await vector_store.delete_document(doc_id)

    # Delete file
    file_path = DOCS_DIR / doc_id
    if file_path.exists():
        file_path.unlink()

    # Return updated document table
    documents = list_documents()
    return document_table(documents)


serve(port=3000)
