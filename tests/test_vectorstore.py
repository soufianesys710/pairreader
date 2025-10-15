"""
Unit tests for pairreader.vectorestore module.

Tests cover:
- VectorStore initialization
- Chunk ingestion
- Query operations
- Sampling and clustering
- Collection management
"""

import pytest

from pairreader.vectorestore import VectorStore

# ============================================================================
# VectorStore Tests
# ============================================================================


class TestVectorStore:
	"""Test suite for VectorStore class."""

	@pytest.mark.unit
	def test_vectorstore_initialization_persistent(self, mock_chromadb_client, mock_chromadb_collection):
		"""Test VectorStore initialization with persistent storage."""
		vs = VectorStore(persistent=True, collection_name="test_collection")

		assert vs.persistent is True
		assert vs.collection_name == "test_collection"
		assert vs.collection is not None

	@pytest.mark.unit
	def test_vectorstore_initialization_ephemeral(self, mock_chromadb_client, mock_chromadb_collection):
		"""Test VectorStore initialization with ephemeral storage."""
		vs = VectorStore(persistent=False, collection_name="test_collection")

		assert vs.persistent is False
		assert vs.collection is not None

	@pytest.mark.unit
	def test_vectorstore_initialization_creates_collection(self, mock_chromadb_client, mock_chromadb_collection):
		"""Test that VectorStore creates collection if it doesn't exist."""
		# Simulate collection not existing
		mock_chromadb_client.get_collection.side_effect = Exception("Collection not found")
		mock_chromadb_client.create_collection.return_value = mock_chromadb_collection

		vs = VectorStore(persistent=False, collection_name="new_collection")

		assert vs.collection is not None
		mock_chromadb_client.create_collection.assert_called_once()

	@pytest.mark.unit
	def test_flush_deletes_and_recreates_collection(self, mock_vectorstore, mock_chromadb_client):
		"""Test that flush() deletes and recreates collection."""
		mock_vectorstore.flush()

		mock_chromadb_client.delete_collection.assert_called_once()
		mock_chromadb_client.create_collection.assert_called_once()

	@pytest.mark.unit
	def test_get_all_ids(self, mock_vectorstore, mock_chromadb_collection):
		"""Test get_all_ids() returns all document IDs."""
		ids = mock_vectorstore.get_all_ids()

		assert isinstance(ids, list)
		assert len(ids) > 0
		mock_chromadb_collection.get.assert_called_once()

	@pytest.mark.unit
	def test_get_all_ids_empty_collection(self, mock_vectorstore, mock_chromadb_collection):
		"""Test get_all_ids() with empty collection."""
		# Simulate empty collection
		mock_chromadb_collection.get.return_value = {"ids": []}

		ids = mock_vectorstore.get_all_ids()

		assert ids == []

	@pytest.mark.unit
	def test_get_len_docs(self, mock_vectorstore, mock_chromadb_collection):
		"""Test get_len_docs() returns document count."""
		length = mock_vectorstore.get_len_docs()

		assert isinstance(length, int)
		assert length > 0

	@pytest.mark.unit
	def test_get_len_docs_empty_collection(self, mock_vectorstore, mock_chromadb_collection):
		"""Test get_len_docs() with empty collection."""
		# Simulate empty collection
		mock_chromadb_collection.get.return_value = {"ids": []}

		length = mock_vectorstore.get_len_docs()

		assert length == 0

	@pytest.mark.unit
	def test_ingest_chunks_without_metadata(self, mock_vectorstore, mock_chromadb_collection):
		"""Test ingest_chunks() without metadata."""
		chunks = ["chunk1", "chunk2", "chunk3"]

		mock_vectorstore.ingest_chunks(chunks)

		mock_chromadb_collection.add.assert_called_once()
		call_args = mock_chromadb_collection.add.call_args
		assert len(call_args.kwargs["ids"]) == 3
		assert call_args.kwargs["documents"] == chunks

	@pytest.mark.unit
	def test_ingest_chunks_with_metadata(self, mock_vectorstore, mock_chromadb_collection):
		"""Test ingest_chunks() with metadata."""
		chunks = ["chunk1", "chunk2"]
		metadatas = [{"filename": "doc1.pdf"}, {"filename": "doc2.pdf"}]

		mock_vectorstore.ingest_chunks(chunks, metadatas=metadatas)

		mock_chromadb_collection.add.assert_called_once()
		call_args = mock_chromadb_collection.add.call_args
		assert call_args.kwargs["metadatas"] == metadatas

	@pytest.mark.unit
	def test_query_with_query_texts(self, mock_vectorstore, mock_chromadb_collection):
		"""Test query() with query texts."""
		results = mock_vectorstore.query(query_texts=["What is ML?"], n_documents=5)

		assert "documents" in results
		assert "metadatas" in results
		assert "ids" in results
		mock_chromadb_collection.query.assert_called_once()

	@pytest.mark.unit
	def test_query_multiple_queries(self, mock_vectorstore, mock_chromadb_collection):
		"""Test query() with multiple query texts."""
		queries = ["What is ML?", "What is AI?"]
		results = mock_vectorstore.query(query_texts=queries, n_documents=10)

		assert results is not None
		mock_chromadb_collection.query.assert_called_once()
		call_args = mock_chromadb_collection.query.call_args
		assert call_args.kwargs["query_texts"] == queries
		assert call_args.kwargs["n_results"] == 10

	@pytest.mark.unit
	def test_query_default_n_documents(self, mock_vectorstore, mock_chromadb_collection):
		"""Test query() with default n_documents."""
		mock_vectorstore.query(query_texts=["test query"])

		call_args = mock_chromadb_collection.query.call_args
		assert call_args.kwargs["n_results"] == 10

	@pytest.mark.unit
	def test_ingest_embedded_chunks_not_implemented(self, mock_vectorstore):
		"""Test that ingest_embedded_chunks() is not implemented."""
		result = mock_vectorstore.ingest_embedded_chunks([])

		# Method exists but doesn't do anything yet
		assert result is None

	@pytest.mark.unit
	def test_collection_attribute(self, mock_vectorstore, mock_chromadb_collection):
		"""Test that collection attribute is accessible."""
		assert mock_vectorstore.collection is mock_chromadb_collection

	@pytest.mark.unit
	def test_db_attribute(self, mock_vectorstore, mock_chromadb_client):
		"""Test that db attribute is accessible."""
		assert mock_vectorstore.db is mock_chromadb_client


# ============================================================================
# VectorStore Sampling and Clustering Tests (Integration)
# TODO: Address these integration tests later
# ============================================================================


class TestVectorStoreSampling:
	"""Test suite for VectorStore sampling and clustering methods."""

	# TODO: Implement proper integration test for sampling
	@pytest.mark.integration
	@pytest.mark.skip(reason="TODO: Integration test - deferred to later")
	def test_get_sample_with_ids(self, mock_vectorstore):
		"""Test get_sample() with valid IDs (if method exists)."""
		pass

	# TODO: Implement proper integration test for consistency
	@pytest.mark.integration
	@pytest.mark.skip(reason="TODO: Integration test - deferred to later")
	def test_get_len_docs_consistency(self, mock_vectorstore):
		"""Test that get_len_docs() matches get_all_ids() length."""
		pass
