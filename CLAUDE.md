# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PairReader is a document chat application built with Chainlit that allows users to upload documents (PDFs, text files) and query them using natural language. It uses a LangGraph-based multi-agent architecture with ChromaDB for vector storage and Docling for document parsing.

## Development Commands

### Running the Application
```bash
uv run pairreader         # Recommended: Uses project.scripts entry point
# OR
uv run chainlit run src/pairreader/__main__.py -w  # Direct Chainlit command with watch mode
```
The application will be available at `http://localhost:8000`

### Installing Dependencies
```bash
uv sync                  # Install all dependencies
uv sync --group dev     # Install with dev dependencies (includes Jupyter)
uv add <package-name>   # Add a new dependency
```

### Environment Setup
Create a `.env` file in the project root with the following variables:
```bash
ANTHROPIC_API_KEY=your_api_key_here
CHAINLIT_AUTH_SECRET=your_secret_here  # Generate with: chainlit create-secret
```

### Development Environment
- Python 3.12+ required
- Uses `uv` package manager (not pip)
- Virtual environment is managed by `uv` in `.venv/`

## Architecture Overview

### LangGraph Workflow
The application uses a LangGraph StateGraph with the following node sequence:
1. **knowledge_base_handler** - Processes Chainlit commands (/Create, /Update) and file uploads
2. **query_optimizer** - Decomposes user queries into subqueries for better retrieval
3. **human_in_the_loop_approver** - Allows user to revise/approve LLM-generated subqueries or request regeneration (with timeout)
4. **info_retriever** - Queries ChromaDB vector store with optimized queries
5. **info_summarizer** - Summarizes retrieved documents using LLM

### Core Components

**PairReaderAgent** (`src/pairreader/agents.py`)
- Main orchestrator that builds and manages the LangGraph workflow
- Uses `InMemorySaver` for checkpointing (not persisted to disk)
- Nodes are stored as tuples `(name, node_instance)` in `self.nodes` list
- `set_params(**params)` propagates settings changes to all nodes (accesses `node[1]` for node instance)
- `route_after_human_in_the_loop_approver()` uses structured output (`HITLDecision`) to route to either `query_optimizer` (regenerate) or `info_retriever` (proceed)

**State Management** (`src/pairreader/schemas.py`)
- `PairReaderState` TypedDict tracks:
  - `messages`: Annotated list with `add_messages` reducer for LLM conversation history
  - `user_query`: Original user question
  - `chainlit_command`: Command type (Create/Update/None)
  - `subqueries`: LLM-generated optimized queries
  - `human_in_the_loop_decision`: HITLDecision with `next_node` field (Literal["query_optimizer", "info_retriever"])
  - `retrieved_documents` & `retrieved_metadatas`: Vector store results
  - `summary`: Final LLM summary
- `HITLDecision` is a Pydantic BaseModel for structured routing decisions

**Node Architecture** (`src/pairreader/nodes.py`)
- All nodes inherit from `ParamsMixin` to support dynamic parameter updates via `set_params(**kwargs)`
- All node `__call__` methods are decorated with `@logging_verbosity` and `@langgraph_stream_verbosity`
- Node classes:
  - `KnowledgeBaseHandler`: Manages file uploads and vector store initialization
  - `QueryOptimizer`: Decomposes queries using LLM (configurable via `query_decomposition` param)
  - `HumanInTheLoopApprover`: Uses `interrupt()` for user interaction, returns structured `HITLDecision`
  - `InfoRetriever`: Queries ChromaDB with subqueries
  - `InfoSummarizer`: Generates final response using LLM

**Utility Decorators** (`src/pairreader/utils.py`)
- `@logging_verbosity`: Logs start/finish of node execution (optional debug param)
- `@langgraph_stream_verbosity`: Writes node status to LangGraph stream using `get_stream_writer()`
- `ParamsMixin`: Base class providing `set_params()` and `get_params()` for dynamic configuration

**Document Processing**
- `DocParser` (`src/pairreader/docparser.py`): Uses Docling's `DocumentConverter` and `HybridChunker`
- `VectorStore` (`src/pairreader/vectorestore.py`): ChromaDB client with persistent storage in `./chroma` directory
- Default collection name: `"knowledge_base"`

**Chainlit Integration** (`src/pairreader/__main__.py`)
- Entry point with `main()` function for CLI command
- Custom `InMemoryDataLayer` for chat history (not persisted between restarts)
- Password authentication: username/password = "admin"/"admin" (TODO: move to secure storage)
- Settings UI for LLM selection, query decomposition toggle, and retrieval parameters
- Uses `cl.context.session.thread_id` for thread-based conversations

### Three Usage Modes
- **Default**: Query existing knowledge base (no command sent)
- **Update** (`/Update`): Add documents to existing knowledge base
- **Create** (`/Create`): Flush knowledge base and start fresh

Commands are triggered via Chainlit commands `/Create` or `/Update`, or via starter buttons in UI

### Key Features

**Query Optimization** (configurable in settings)
- `query_decomposition`: Breaks complex queries into sub-queries using LLM
- User can revise/approve subqueries or request regeneration via human-in-the-loop

**Document Ingestion**
- Accepts PDF and text files
- Max 5 files, 10MB each
- Files are chunked and stored with metadata (filename)
- Chunks are contextualized using `HybridChunker.contextualize()`

**Interrupts and Timeouts**
- File upload prompts have 90s timeout
- Uses LangGraph `interrupt()` for user interaction
- When timeout occurs, user can continue with existing knowledge base

## Important Implementation Details

### LLM Configuration
- Default LLM: `anthropic:claude-3-5-haiku-latest`
- Fallback LLM: `anthropic:claude-3-7-sonnet-latest`
- LLMs are initialized using `langchain.chat_models.init_chat_model`
- Fallback is configured with `.with_fallbacks([...])`
- Each node that uses LLM has an `llm` property that returns the configured LLM with fallback

### Vector Store Behavior
- `/Create` command calls `vectorstore.flush()` (deletes and recreates collection)
- `/Update` command appends to existing collection
- Query results include top `n_documents` (default: 10, configurable via settings)
- Uses default ChromaDB embedding (all-MiniLM-L6-v2)
- Persistent storage in `./chroma` directory

### Streaming and Execution
- LangGraph workflow is invoked with `await pairreader(input=input, config=config)` in `on_message()`
- Config includes thread_id: `{"configurable": {"thread_id": thread_id}}`
- Nodes use LangGraph's `get_stream_writer()` to send status updates
- LLM message history is maintained in `state["messages"]` with `add_messages` reducer

### Session Management
- Thread-based conversation using `cl.context.session.thread_id`
- Config passed to workflow: `{"configurable": {"thread_id": thread_id}}`
- InMemoryDataLayer handles chat history (not persisted between restarts)

### Authentication
- Uses `@cl.password_auth_callback`
- Current credentials hardcoded (admin/admin) - TODO: use database
- See Environment Setup section for required environment variables

## Code Style and Patterns

### Node Implementation Pattern
All LangGraph nodes follow this pattern:
```python
class NodeName(ParamsMixin):
    def __init__(self, param1: type = default, ...):
        self.param1 = param1
        ...

    @property
    def llm(self):  # If node uses LLM
        return (
            init_chat_model(self.llm_name)
            .with_fallbacks([init_chat_model(self.fallback_llm_name)])
        )

    @logging_verbosity
    @langgraph_stream_verbosity
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict:
        # Node logic here
        return {"state_key": value}
```

### Important Patterns
- Node classes are instantiated as tuples in `PairReaderAgent.__init__`: `("node_name", NodeInstance())`
- When accessing nodes from `self.nodes` list, use `node[1]` to get the instance (e.g., in `set_params()`)
- LLM nodes use a `@property` for `llm` to ensure fresh initialization with current params
- State updates return a dict with only the keys being updated
- Use `interrupt()` from `langgraph.types` to pause workflow for user input
- Structured outputs use Pydantic BaseModel (see `HITLDecision`)

### Error Handling
- Use try/except with logger.error() for error handling (see `docparser.py`)
- ChromaDB queries support `where_document` filters (contains/not_contains)

## Package Structure
- Entry point: `src/pairreader/__main__.py` with `main()` function
- Package script defined in `pyproject.toml`: `pairreader = "pairreader.__main__:main"`
- This allows running via `uv run pairreader` or just `pairreader` after installation

## Future TODOs (from codebase comments)
- Enhanced table and image extraction from Docling
- Embedding and tokenization-aware chunking
- Retrieve chunks with metadata (e.g., page numbers)
- Validate distance metric compatibility with embedding model
- Move authentication to database with hashed passwords
- Explore OAuth and header-based authentication
