# Milvus vector database integration for autocare-backend
# This module provides a handler for connecting to and operating on a local Milvus instance.
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from typing import List, Dict, Any, Optional

class MilvusHandler:
    def __init__(self, host: str = "localhost", port: str = "19530", collection_name: str = "Groundknowledge"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self._connect()
        self._ensure_collection()

    def _connect(self):
        connections.connect(alias="default", host=self.host, port=self.port)

    def _ensure_collection(self):
        if self.collection_name not in [c for c in Collection.list()]:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
                FieldSchema(name="book_title", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="content_chunk", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),  # Changed to 1024 dimensions
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            schema = CollectionSchema(fields, description="Ground knowledge chunks")
            Collection(self.collection_name, schema)
        self.collection = Collection(self.collection_name)

    def insert(self, data: List[Dict[str, Any]]):
        # Data should be a list of dicts with keys matching the schema
        fields = ["id", "book_title", "content_chunk", "vector", "page_number", "metadata"]
        insert_data = [[d.get(f) for d in data] for f in fields]
        self.collection.insert(insert_data)

    def search(self, query_vector: List[float], top_k: int = 5, filter_expr: Optional[str] = None):
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["id", "book_title", "content_chunk", "page_number", "metadata"]
        )
        return results

    def query(self, expr=None, output_fields=None, limit=100):
        """
        Query the collection for documents (non-vector search).
        Args:
            expr: Optional filter expression.
            output_fields: List of fields to return.
            limit: Max number of results.
        Returns:
            List of dicts for each entity.
        """
        results = self.collection.query(expr=expr, output_fields=output_fields, limit=limit)
        return results

    def count(self):
        return self.collection.num_entities
