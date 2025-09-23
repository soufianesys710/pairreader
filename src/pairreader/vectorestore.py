# TODO: make sure the distance metric is supported by the embedding model


from pairreader.docparser import DocParser
import chromadb
import uuid

from typing import Any, Dict, List, Optional, Union
import logging

class VectorStore:
    """
    Handles storage and retrieval of document chunks in a ChromaDB collection.

    Args:
        persistent (bool): Whether to use persistent ChromaDB storage.
        collection_name (str): Name of the ChromaDB collection.

    Example:
        vs = VectorStore(file='mydoc.pdf')
        vs.ingest_chunks()  # Ingests and stores chunks in ChromaDB
        results = vs.query(query_texts=['What is X?'])
    """

    def __init__(
        self, persistent: bool = True, collection_name: str = "knowledge_base"
    ):
        self.persistent = persistent
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)
        try:
            if persistent:
                self.db = chromadb.PersistentClient()
            else:
                self.db = chromadb.EphemeralClient()
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
        try:
            self.collection = self.db.get_collection(collection_name)
        except Exception as e:
            self.logger.warning(f"{collection_name} collection doesn't exist, creating it! Error: {e}")
            try:
                self.collection = self.db.create_collection(collection_name)
            except Exception as ce:
                self.logger.error(f"Failed to create collection {collection_name}: {ce}")
                raise

    def flush(self):
        self.db.delete_collection(self.collection_name)
        self.collection = self.db.create_collection(self.collection_name)

    def ingest_chunks(self, chunks: List[str], metadatas: Optional[Dict] = None) -> None:
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        self.collection.add(
            ids=ids, documents=chunks, metadatas=metadatas
        )

    def ingest_embedded_chunks(self, embedded_chunks: List[Dict], metdatas: Optional[Dict] = None) -> None:
        pass

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        contains: Optional[List[str]] = None,
        not_contains: Optional[List[str]] = None,
        n_results: int = 10,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Query the ChromaDB collection.
        Args:
            query_texts: List of query strings.
            contains: List of strings that must be contained in the document.
            not_contains: List of strings that must not be contained in the document.
            n_results: Number of results to return.
            **kwargs: Additional filter arguments for 'where'.
        Returns:
            Query results as a dictionary.
        """
        # Base args
        query_args = {
            "query_texts": query_texts,
            "n_results": n_results,
        }
        # conditions
        where_document = [
            *[{"$contains": term} for term in contains or []],
            *[{"$not_contains": term} for term in not_contains or []]
        ]
        if where_document:
            query_args["where_document"] = (
                where_document[0] if len(where_document) == 1 
                else {"$or": where_document}
            )
        if kwargs:
            query_args["where"] = kwargs
        # query
        results = self.collection.query(**query_args)
        return results
