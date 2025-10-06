# TODO: make sure the distance metric is supported by the embedding model


from pairreader.docparser import DocParser
import chromadb
import uuid
import random
import asyncio

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
        self, 
        persistent: bool = True,
        path: str = "./chroma",
        collection_name: str = "knowledge_base",
    ):
        self.persistent = persistent
        self.path = path
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)
        try:
            if persistent:
                self.db = chromadb.PersistentClient(path=path)
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

    def get_all_ids(self) -> List[str]:
        """Get all document IDs from the collection."""
        self.all_ids =  self.collection.get()["ids"]
        if not self.all_ids:
            self.logger.warning("Collection is empty, returning empty sample")
            self.all_ids = []
            return []
        else:
            return self.all_ids

    def get_len_docs(self) -> int:
        """Get the total number of documents in the collection."""
        self.len_docs = len(self.get_all_ids())
        if self.len_docs == 0:
            self.logger.warning("Collection is empty, returning empty clusters")
        return self.len_docs

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
        n_documents: int = 10,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Query the ChromaDB collection.
        Args:
            query_texts: List of query strings.
            contains: List of strings that must be contained in the document.
            not_contains: List of strings that must not be contained in the document.
            n_documents: Number of results to return.
            **kwargs: Additional filter arguments for 'where'.
        Returns:
            Query results as a dictionary.
        """
        # Base args
        query_args = {
            "query_texts": query_texts,
            "n_results": n_documents,
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

    def get_sample(self, percentage: float) -> List[str]:
        """
        Sample a percentage of document IDs from the collection.

        Args:
            percentage: Percentage of documents to sample (0.0 to 1.0).

        Returns:
            List of sampled document IDs.
        """
        if not 0.0 <= percentage <= 1.0:
            raise ValueError("Percentage must be between 0.0 and 1.0")

        # Get all document IDs and Calculate sample size
        all_ids = self.get_all_ids()
        sample_size = max(1, int(len(all_ids) * percentage))

        # Random sample
        sampled_ids = random.sample(all_ids, sample_size)
        self.logger.info(f"Sampled {len(sampled_ids)} documents ({percentage*100:.1f}%) from {len(all_ids)} total")

        return sampled_ids

    async def _query_cluster(self, sample_id: str, cluster_size: int, cluster_idx: int) -> Optional[List[str]]:
        """
        Query a single cluster asynchronously.

        Args:
            sample_id: Document ID to use as cluster center.
            cluster_size: Number of documents to retrieve.
            cluster_idx: Index of the cluster for logging.

        Returns:
            List of document IDs in the cluster, or None if query failed.
        """
        self.logger.debug(f"Cluster {cluster_idx}: Starting query for sample ID {sample_id}")

        # Get the sample document text
        sample_doc = self.collection.get(ids=[sample_id])

        if not sample_doc["documents"] or not sample_doc["documents"][0]:
            self.logger.warning(f"Cluster {cluster_idx}: Sample ID {sample_id} not found or has no text, skipping")
            return None

        sample_text = sample_doc["documents"][0]

        # Query using the sample document as query text
        results = self.collection.query(
            query_texts=[sample_text],
            n_results=cluster_size
        )

        # Extract IDs from results
        if results["ids"] and results["ids"][0]:
            cluster_ids = results["ids"][0]
            self.logger.info(f"Cluster {cluster_idx}: Retrieved {len(cluster_ids)} documents for sample ID {sample_id}")
            return cluster_ids
        else:
            self.logger.warning(f"Cluster {cluster_idx}: No results found for sample ID {sample_id}")
            return None

    async def get_clusters(self, sample_ids: List[str], percentage: float) -> List[List[str]]:
        """
        Create semantic clusters by using sample documents as query points.
        Queries are performed in parallel for efficiency.

        Args:
            sample_ids: List of document IDs to use as cluster centers.
            percentage: Percentage of total documents to retrieve per cluster (0.0 to 1.0).

        Returns:
            List of clusters, where each cluster is a list of document IDs.
        """
        if not 0.0 <= percentage <= 1.0:
            raise ValueError("Percentage must be between 0.0 and 1.0")

        # Get total document count and calculate cluster size
        total_docs = self.get_len_docs()
        cluster_size = max(1, int(total_docs * percentage))

        self.logger.info(f"Building {len(sample_ids)} clusters with size {cluster_size} each")

        # Build clusters in parallel
        tasks = [
            self._query_cluster(sample_id, cluster_size, idx)
            for idx, sample_id in enumerate(sample_ids)
        ]

        results = await asyncio.gather(*tasks)

        # Filter out None results and collect all cluster IDs
        clusters = [cluster for cluster in results if cluster is not None]
        clustered_ids = set()
        for cluster in clusters:
            clustered_ids.update(cluster)

        # Calculate and log orphan percentage
        orphan_count = total_docs - len(clustered_ids)
        orphan_percentage = (orphan_count / total_docs) * 100 if total_docs > 0 else 0

        self.logger.info(
            f"Created {len(clusters)} clusters covering {len(clustered_ids)} documents. "
            f"Orphan documents: {orphan_percentage:.1f}% ({orphan_count}/{total_docs}) not included in any cluster"
        )

        return clusters
