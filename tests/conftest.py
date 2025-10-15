"""
Shared pytest fixtures and configuration for PairReader tests.

This module provides common fixtures that can be used across all test files.
"""

import asyncio
import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# ============================================================================
# Pytest Configuration
# ============================================================================


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for async tests."""
    return asyncio.get_event_loop_policy()


# ============================================================================
# State and Schema Fixtures
# ============================================================================


@pytest.fixture
def empty_state() -> dict[str, Any]:
    """Create an empty PairReaderState for testing."""
    return {
        "messages": [],
        "user_query": None,
        "user_command": None,
        "subqueries": None,
        "human_in_the_loop_decision": None,
        "retrieved_documents": None,
        "retrieved_metadatas": None,
        "summary": None,
        "clusters": None,
        "cluster_summaries": None,
        "summary_of_summaries": None,
    }


@pytest.fixture
def sample_state() -> dict[str, Any]:
    """Create a sample PairReaderState with some data."""
    return {
        "messages": [
            HumanMessage(content="What is machine learning?"),
            AIMessage(content="Machine learning is a subset of artificial intelligence..."),
        ],
        "user_query": "What is machine learning?",
        "user_command": None,
        "subqueries": ["What is machine learning?", "What are types of machine learning?"],
        "human_in_the_loop_decision": None,
        "retrieved_documents": ["Document 1 content", "Document 2 content"],
        "retrieved_metadatas": [{"filename": "doc1.pdf"}, {"filename": "doc2.pdf"}],
        "summary": "Machine learning is a field of AI that enables computers to learn from data.",
        "clusters": None,
        "cluster_summaries": None,
        "summary_of_summaries": None,
    }


@pytest.fixture
def sample_messages() -> list:
    """Create sample LangChain messages for testing."""
    return [
        HumanMessage(content="Hello, how are you?"),
        AIMessage(content="I'm doing well, thank you!"),
        HumanMessage(content="Can you help me with a question?"),
        AIMessage(content="Of course! What would you like to know?"),
    ]


# ============================================================================
# Mock Chainlit Fixtures
# ============================================================================


@pytest.fixture
def mock_chainlit(monkeypatch):
    """Mock Chainlit user I/O operations."""
    mock_cl = MagicMock()

    # Mock Message
    mock_message = AsyncMock()
    mock_message.send = AsyncMock()
    mock_message.stream_token = AsyncMock()
    mock_message.update = AsyncMock()
    mock_message.content = "Mocked response"
    mock_cl.Message = Mock(return_value=mock_message)

    # Mock AskUserMessage
    mock_ask_user = AsyncMock()
    mock_ask_user.send = AsyncMock(return_value={"output": "User response"})
    mock_cl.AskUserMessage = Mock(return_value=mock_ask_user)

    # Mock AskFileMessage
    mock_ask_file = AsyncMock()
    mock_file = MagicMock()
    mock_file.path = "/tmp/test_document.pdf"
    mock_file.name = "test_document.pdf"
    mock_ask_file.send = AsyncMock(return_value=[mock_file])
    mock_cl.AskFileMessage = Mock(return_value=mock_ask_file)

    # Patch chainlit module
    import sys

    sys.modules["chainlit"] = mock_cl
    monkeypatch.setitem(sys.modules, "chainlit", mock_cl)

    return mock_cl


# ============================================================================
# Mock LLM Fixtures
# ============================================================================


@pytest.fixture
def mock_llm():
    """Mock LangChain LLM with predefined responses."""
    mock = AsyncMock()
    mock.ainvoke = AsyncMock(return_value=AIMessage(content="Mocked LLM response"))
    mock.astream = AsyncMock()

    async def mock_stream(*args, **kwargs):
        """Simulate streaming chunks."""
        chunks = [
            MagicMock(content="Mocked "),
            MagicMock(content="streaming "),
            MagicMock(content="response"),
        ]
        for chunk in chunks:
            yield chunk

    mock.astream.return_value = mock_stream()
    mock.invoke = Mock(return_value=AIMessage(content="Mocked LLM response"))
    mock.bind_tools = Mock(return_value=mock)
    mock.with_structured_output = Mock(return_value=mock)
    mock.with_fallbacks = Mock(return_value=mock)

    return mock


@pytest.fixture
def mock_init_chat_model(monkeypatch, mock_llm):
    """Mock LangChain's init_chat_model function."""

    def mock_init(*args, **kwargs):
        return mock_llm

    monkeypatch.setattr("langchain.chat_models.init_chat_model", mock_init)
    return mock_init


# ============================================================================
# Mock ChromaDB Fixtures
# ============================================================================


@pytest.fixture
def mock_chromadb_collection():
    """Mock ChromaDB collection for testing."""
    mock_collection = MagicMock()
    mock_collection.add = Mock()
    mock_collection.query = Mock(
        return_value={
            "ids": [["id1", "id2", "id3"]],
            "documents": [["Document 1", "Document 2", "Document 3"]],
            "metadatas": [
                [{"filename": "doc1.pdf"}, {"filename": "doc2.pdf"}, {"filename": "doc3.pdf"}]
            ],
            "distances": [[0.1, 0.2, 0.3]],
        }
    )
    mock_collection.get = Mock(return_value={"ids": ["id1", "id2", "id3", "id4", "id5"]})
    mock_collection.count = Mock(return_value=5)
    mock_collection.delete = Mock()

    return mock_collection


@pytest.fixture
def mock_chromadb_client(mock_chromadb_collection, monkeypatch):
    """Mock ChromaDB client."""
    mock_client = MagicMock()
    mock_client.get_collection = Mock(return_value=mock_chromadb_collection)
    mock_client.create_collection = Mock(return_value=mock_chromadb_collection)
    mock_client.delete_collection = Mock()

    # Mock chromadb module
    mock_chromadb = MagicMock()
    mock_chromadb.PersistentClient = Mock(return_value=mock_client)
    mock_chromadb.EphemeralClient = Mock(return_value=mock_client)

    monkeypatch.setattr("chromadb.PersistentClient", mock_chromadb.PersistentClient)
    monkeypatch.setattr("chromadb.EphemeralClient", mock_chromadb.EphemeralClient)

    return mock_client


# ============================================================================
# Mock Docling Fixtures
# ============================================================================


@pytest.fixture
def mock_docling_document():
    """Mock Docling document object."""
    mock_doc = MagicMock()
    mock_doc.document = MagicMock()
    return mock_doc


@pytest.fixture
def mock_document_converter(mock_docling_document, monkeypatch):
    """Mock Docling DocumentConverter."""
    mock_converter = MagicMock()
    mock_converter.convert = Mock(return_value=mock_docling_document)

    # Mock the DocumentConverter class
    mock_dc_class = Mock(return_value=mock_converter)
    monkeypatch.setattr("docling.document_converter.DocumentConverter", mock_dc_class)

    return mock_converter


@pytest.fixture
def mock_hybrid_chunker(monkeypatch):
    """Mock Docling HybridChunker."""
    mock_chunker = MagicMock()

    # Create mock chunks
    mock_chunks = [
        MagicMock(text="Chunk 1 content", meta={"page": 1}),
        MagicMock(text="Chunk 2 content", meta={"page": 1}),
        MagicMock(text="Chunk 3 content", meta={"page": 2}),
    ]

    mock_chunker.chunk = Mock(return_value=iter(mock_chunks))
    mock_chunker.contextualize = Mock(side_effect=lambda chunk: f"Contextualized: {chunk.text}")

    # Mock the HybridChunker class
    mock_hc_class = Mock(return_value=mock_chunker)
    monkeypatch.setattr("docling.chunking.HybridChunker", mock_hc_class)

    return mock_chunker


# ============================================================================
# File System Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing."""
    file_path = os.path.join(temp_dir, "sample.txt")
    with open(file_path, "w") as f:
        f.write("This is a sample text file.\n")
        f.write("It contains multiple lines.\n")
        f.write("Used for testing document parsing.\n")
    return file_path


@pytest.fixture
def sample_pdf_path(temp_dir):
    """Return a mock PDF file path (doesn't create actual PDF)."""
    return os.path.join(temp_dir, "sample.pdf")


# ============================================================================
# VectorStore Fixtures
# ============================================================================


@pytest.fixture
def mock_vectorstore(mock_chromadb_client, mock_chromadb_collection):
    """Create a mock VectorStore instance."""
    from pairreader.vectorestore import VectorStore

    # Create VectorStore with mocked ChromaDB
    vs = VectorStore(persistent=False, collection_name="test_collection")
    vs.collection = mock_chromadb_collection
    vs.db = mock_chromadb_client

    return vs


# ============================================================================
# DocParser Fixtures
# ============================================================================


@pytest.fixture
def mock_docparser(mock_document_converter, mock_hybrid_chunker):
    """Create a mock DocParser instance."""
    from pairreader.docparser import DocParser

    parser = DocParser(converter=mock_document_converter, chunker=mock_hybrid_chunker)
    return parser


# ============================================================================
# Node Parameter Fixtures
# ============================================================================


@pytest.fixture
def default_node_params() -> dict[str, Any]:
    """Default parameters for node initialization."""
    return {
        "llm_name": "anthropic:claude-3-5-haiku-latest",
        "fallback_llm_name": "anthropic:claude-3-7-sonnet-latest",
        "n_documents": 10,
        "query_decomposition": True,
        "n_sample": 0,
        "p_sample": 0.3,
        "cluster_percentage": 0.5,
        "min_cluster_size": 0,
        "max_cluster_size": 0,
    }
