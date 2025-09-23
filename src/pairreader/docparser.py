# for more info on docling rich parsing and chunking and serialization
# concepts: https://docling-project.github.io/docling/concepts/
# examples with code: https://docling-project.github.io/docling/examples/
# TODO: more in depth table and image information extraction techniques from docling
# TODO: embedding and tokenization aware chunking
# TODO: retrive chunk with metadata when possible e.g. page in a pdf file
# TODO: make sure the distance metric is supported by the embedding model

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import os
import logging

class DocParser:
    """
    Parses a single document file, converts it using a DocumentConverter,
    and chunks it using a HybridChunker.

    Args:
        converter (DocumentConverter, optional): Converter for document formats.
        chunker (HybridChunker, optional): Chunker for splitting documents.

    Example:
        parser = DocParser()
        parser.parse(file="mydoc.pdf")
        chunks = parser.get_chunks()
    """

    def __init__(self, converter=None, chunker=None):
        self.doc = None
        self.converter = converter if converter is not None else DocumentConverter()
        self.chunker = chunker if chunker is not None else HybridChunker()
        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            logging.basicConfig(level=logging.INFO)

    def parse(self, file: str):
        self.file = file
        self.file_path = os.path.abspath(self.file)
        self.file_name = os.path.basename(self.file)
        self.file_extension = os.path.splitext(self.file)[1]
        try:
            self.doc = self.converter.convert(self.file)
        except Exception as e:
            self.logger.error(f"Failed to process file '{self.file}': {e}")
            self.doc = None

    def get_chunks(self):
        if self.doc is None:
            self.logger.error("No document loaded. Call parse() first.")
            return []
        try:
            chunks = list(self.chunker.chunk(self.doc.document))
            contextualized_chunks = [self.chunker.contextualize(chunk) for chunk in chunks]
            return contextualized_chunks
        except Exception as e:
            self.logger.error(f"Failed to chunk document '{self.file}': {e}")
            return []

    def get_embedded_chunks(self):
        pass

    def get_chunk(self, index):
        chunks = self.get_chunks()
        if index < 0 or index >= len(chunks):
            self.logger.error(f"Chunk index {index} out of range.")
            return None
        else:
            return chunks[index]
        
    def get_embedded_chunk(self, index):
        pass

    def save_as_markdown(self, out_path):
        if self.doc is not None:
            self.doc.save_as_markdown(out_path)