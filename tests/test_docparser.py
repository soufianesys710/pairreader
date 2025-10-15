"""
Unit tests for pairreader.docparser module.

Tests cover:
- DocParser initialization
- Document parsing
- Chunk extraction
- Error handling
"""

import os
from unittest.mock import Mock, patch

import pytest

from pairreader.docparser import DocParser

# ============================================================================
# DocParser Tests
# ============================================================================


class TestDocParser:
    """Test suite for DocParser class."""

    @pytest.mark.unit
    def test_docparser_initialization_defaults(self):
        """Test DocParser initialization with default converter and chunker."""
        with (
            patch("pairreader.docparser.DocumentConverter"),
            patch("pairreader.docparser.HybridChunker"),
        ):
            parser = DocParser()

            assert parser.doc is None
            assert parser.converter is not None
            assert parser.chunker is not None

    @pytest.mark.unit
    def test_docparser_initialization_custom(self):
        """Test DocParser initialization with custom converter and chunker."""
        mock_converter = Mock()
        mock_chunker = Mock()

        parser = DocParser(converter=mock_converter, chunker=mock_chunker)

        assert parser.converter is mock_converter
        assert parser.chunker is mock_chunker

    @pytest.mark.unit
    def test_parse_success(self, mock_docparser, sample_text_file):
        """Test successful document parsing."""
        parser = mock_docparser
        parser.parse(sample_text_file)

        assert parser.file == sample_text_file
        assert parser.file_name == os.path.basename(sample_text_file)
        assert parser.file_extension == ".txt"
        assert parser.doc is not None

    @pytest.mark.unit
    def test_parse_file_path_attributes(self, mock_docparser, sample_text_file):
        """Test that parse() sets correct file path attributes."""
        parser = mock_docparser
        parser.parse(sample_text_file)

        assert parser.file_path == os.path.abspath(sample_text_file)
        assert parser.file == sample_text_file
        assert os.path.isabs(parser.file_path)

    @pytest.mark.unit
    def test_parse_error_handling(self, mock_document_converter):
        """Test parse() error handling when converter fails."""
        # Make converter raise an exception
        mock_document_converter.convert.side_effect = Exception("Parse error")

        parser = DocParser(converter=mock_document_converter)
        parser.parse("nonexistent.pdf")

        # Should not raise, but doc should be None
        assert parser.doc is None

    @pytest.mark.unit
    def test_get_chunks_success(self, mock_docparser, sample_text_file):
        """Test successful chunk extraction."""
        parser = mock_docparser
        parser.parse(sample_text_file)

        chunks = parser.get_chunks()

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    @pytest.mark.unit
    def test_get_chunks_no_document_loaded(self):
        """Test get_chunks() when no document is loaded."""
        with (
            patch("pairreader.docparser.DocumentConverter"),
            patch("pairreader.docparser.HybridChunker"),
        ):
            parser = DocParser()

            chunks = parser.get_chunks()

            assert chunks == []

    @pytest.mark.unit
    def test_get_chunks_error_handling(self, mock_docparser, mock_hybrid_chunker, sample_text_file):
        """Test get_chunks() error handling when chunking fails."""
        parser = mock_docparser
        parser.parse(sample_text_file)

        # Make chunker raise an exception
        mock_hybrid_chunker.chunk.side_effect = Exception("Chunking error")

        chunks = parser.get_chunks()

        # Should return empty list on error
        assert chunks == []

    @pytest.mark.unit
    def test_get_chunk_valid_index(self, mock_docparser, sample_text_file):
        """Test get_chunk() with valid index."""
        parser = mock_docparser
        parser.parse(sample_text_file)

        chunk = parser.get_chunk(0)

        assert chunk is not None
        assert isinstance(chunk, str)

    @pytest.mark.unit
    def test_get_chunk_invalid_index_negative(self, mock_docparser, sample_text_file):
        """Test get_chunk() with negative index."""
        parser = mock_docparser
        parser.parse(sample_text_file)

        chunk = parser.get_chunk(-1)

        assert chunk is None

    @pytest.mark.unit
    def test_get_chunk_invalid_index_out_of_range(self, mock_docparser, sample_text_file):
        """Test get_chunk() with out-of-range index."""
        parser = mock_docparser
        parser.parse(sample_text_file)

        chunk = parser.get_chunk(999)

        assert chunk is None

    @pytest.mark.unit
    def test_chunk_contextualization(self, mock_docparser, mock_hybrid_chunker, sample_text_file):
        """Test that chunks are contextualized."""
        parser = mock_docparser
        parser.parse(sample_text_file)

        chunks = parser.get_chunks()

        # Verify contextualize was called for each chunk
        assert mock_hybrid_chunker.contextualize.called
        # Verify chunks are contextualized (contain prefix)
        assert all("Contextualized:" in chunk for chunk in chunks)

    @pytest.mark.unit
    def test_get_embedded_chunks_not_implemented(self, mock_docparser):
        """Test that get_embedded_chunks() is not implemented."""
        parser = mock_docparser

        # Method exists but doesn't return anything
        result = parser.get_embedded_chunks()

        assert result is None

    @pytest.mark.unit
    def test_get_embedded_chunk_not_implemented(self, mock_docparser):
        """Test that get_embedded_chunk() is not implemented."""
        parser = mock_docparser

        # Method exists but doesn't return anything
        result = parser.get_embedded_chunk(0)

        assert result is None

    @pytest.mark.unit
    def test_parse_pdf_file(self, mock_docparser, sample_pdf_path):
        """Test parsing a PDF file."""
        parser = mock_docparser
        parser.parse(sample_pdf_path)

        assert parser.file_extension == ".pdf"
        assert parser.doc is not None

    @pytest.mark.unit
    def test_multiple_parses(self, mock_docparser, sample_text_file):
        """Test that parser can parse multiple documents."""
        parser = mock_docparser

        # Parse first document
        parser.parse(sample_text_file)
        first_doc = parser.doc
        first_filename = parser.file_name

        # Parse second document (reuse parser)
        parser.parse(sample_text_file)
        second_doc = parser.doc
        second_filename = parser.file_name

        assert first_filename == second_filename
        assert first_doc is not None
        assert second_doc is not None
