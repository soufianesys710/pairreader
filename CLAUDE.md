# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PairReader is a document chat application built with Chainlit that allows users to upload documents (PDFs, text files) and query them using natural language. It uses a LangGraph-based multi-agent architecture with ChromaDB for vector storage and Docling for document parsing.

## Development Commands

### Running the Application
```bash
uv run chainlit run app.py -w
```
The application will be available at `http://localhost:8000`

### Installing Dependencies
```bash
uv sync                  # Install all dependencies
uv sync --group dev     # Install with dev dependencies (includes Jupyter)
uv add <package-name>   # Add a new dependency
```

### Development Environment
- Python 3.12+ required
- Uses `uv` package manager (not pip)
- Virtual environment is managed by `uv` in `.venv/`

## Architecture Overview

### LangGraph Workflow
The application uses a LangGraph StateGraph with the following node sequence:
1. **chainlit_command_handler** - Processes Chainlit commands (/Create, /Update) and file uploads
2. **query_optimizer** - Decomposes and expands user queries for better retrieval
3. **human_reviser** - Allows user to revise LLM-generated subqueries (with 60s timeout)
4. **info_retriever** - Queries ChromaDB vector store with optimized queries
5. **info_summarizer** - Summarizes retrieved documents using LLM

### Core Components

**PairReaderAgent** (`src/pairreader/agents.py`)
- Main orchestrator that builds and manages the LangGraph workflow
- Uses `InMemorySaver` for checkpointing (not persisted to disk)
- Manages all nodes and state transitions

**State Management** (`src/pairreader/schemas.py`)
- `PairReaderState` TypedDict tracks:
  - `user_query`: Original user question
  - `chainlit_command`: Command type (Create/Update/None)
  - `llm_subqueries`: LLM-generated optimized queries
  - `human_subqueries`: User-revised queries (or LLM queries if timeout)
  - `retrieved_documents` & `retrieved_metadatas`: Vector store results
  - `response`: Final LLM summary

**Document Processing**
- `DocParser` (`src/pairreader/docparser.py`): Uses Docling's `DocumentConverter` and `HybridChunker`
- `VectorStore` (`src/pairreader/vectorestore.py`): ChromaDB client with persistent storage
- Default collection name: `"knowledge_base"`

**Chainlit Integration** (`app.py`)
- Custom `InMemoryDataLayer` for chat history (not persisted)
- Password authentication: username/password = "admin"/"admin" (TODO: move to secure storage)
- Settings UI for LLM selection, query optimization toggles, and retrieval parameters

### Three Usage Modes
- **Dafault**: Query existing knowledge base
- **Update**: Add documents to existing knowledge base
- **Create**: Flush knowledge base and start fresh

Commands are triggered via Chainlit commands `/Create` or `/Update`, or via starter buttons

### Key Features

**Query Optimization** (configurable in settings)
- `query_decomposition`: Breaks complex queries into sub-queries
- `query_expansion`: Generates synonymous variations (requires decomposition enabled)
- `max_expansion`: Limits expansion queries (default: 7)

**Document Ingestion**
- Accepts PDF and text files
- Max 5 files, 10MB each
- Files are chunked and stored with metadata (filename)
- Chunks are contextualized using `HybridChunker.contextualize()`

**Interrupts and Timeouts**
- File upload prompts have 60s timeout
- Subquery revision prompts have 60s timeout
- Uses LangGraph `interrupt()` for user interaction

## Important Implementation Details

### LLM Configuration
- Default LLM: `anthropic:claude-3-5-haiku-latest`
- Fallback LLM: `anthropic:claude-3-7-sonnet-latest`
- LLMs are initialized using `langchain.chat_models.init_chat_model`
- Fallback is configured with `.with_fallbacks()`

### Vector Store Behavior
- `/Create` command calls `vectorstore.flush()` (deletes and recreates collection)
- `/Update` command appends to existing collection
- Query results include top `n_documents` (default: 10, configurable)
- Uses default ChromaDB embedding (all-MiniLM-L6-v2)

### Streaming
- LangGraph workflow streams in two modes: `["messages", "updates"]`
- Only messages from `info_summarizer` node are displayed to user
- Interrupts are caught and displayed from `updates` stream

### Session Management
- Thread-based conversation using `cl.context.session.thread_id`
- Config passed to workflow: `{"configurable": {"thread_id": thread_id}}`
- InMemoryDataLayer handles chat history (not persisted between restarts)

### Authentication
- Uses `@cl.password_auth_callback`
- `CHAINLIT_AUTH_SECRET` must be set (generate with `chainlit create-secret`)
- Current credentials hardcoded (admin/admin) - TODO: use database

### Environment Variables
Check `.env` file for required API keys (likely ANTHROPIC_API_KEY)

## Code Style Notes

- All LangGraph nodes are async functions decorated with `@logging_verbosity` and `@langgraph_stream_verbosity`
- Nodes inherit from `ParamsMixin` for dynamic parameter updates from Chainlit settings
- Error handling uses try/except with logger.error() (see `docparser.py`)
- ChromaDB queries support `where_document` filters (contains/not_contains)

## Future TODOs (from codebase comments)
- Enhanced table and image extraction from Docling
- Embedding and tokenization-aware chunking
- Retrieve chunks with metadata (e.g., page numbers)
- Validate distance metric compatibility with embedding model
- Move authentication to database with hashed passwords
- Explore OAuth and header-based authentication
