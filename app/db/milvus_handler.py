# Milvus vector database integration for autocare-backend
# This module provides a handler for connecting to and operating on a local Milvus instance.
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from typing import List, Dict, Any, Optional
import os
from app.core.interfaces import Protocol

class IMilvusHandler(Protocol):
    def insert(self, data: List[Dict[str, Any]]): ...
    # Add other method signatures as needed

class MilvusHandler(IMilvusHandler):
    def __init__(self, host: str = None, port: str = None, collection_name: str = "Groundknowledge"):
        # Read from environment variables, fallback to defaults
        self.host = host or os.getenv("MILVUS_HOST", "localhost")  # Changed default from 'milvus' to 'localhost'
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        self.collection_name = collection_name
        self.car_collection_name = "CarManuals"
        self._connect()
        self._ensure_collection()
        self._ensure_car_collection()

    def _connect(self):
        connections.connect(alias="default", host=self.host, port=self.port)

    def _ensure_collection(self):
        if self.collection_name not in utility.list_collections():
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
                FieldSchema(name="book_title", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="content_chunk", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),  # Changed to 768 dimensions
                FieldSchema(name="page_number", dtype=DataType.INT64),                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            schema = CollectionSchema(fields, description="Ground knowledge chunks")
            Collection(self.collection_name, schema)
        self.collection = Collection(self.collection_name)
        # Ensure index exists on the vector field
        if "vector" in [f.name for f in self.collection.schema.fields]:
            if not self.collection.indexes:
                self.collection.create_index(
                    field_name="vector",
                    index_params={
                        "index_type": "IVF_FLAT",  # or "HNSW", "IVF_SQ8", etc.
                        "metric_type": "L2",
                        "params": {"nlist": 128}
                    }
                )

    def _ensure_car_collection(self):
        """Ensure car manuals collection exists"""
        if self.car_collection_name not in utility.list_collections():
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
                FieldSchema(name="car_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="make", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="model", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="year", dtype=DataType.INT32),
                FieldSchema(name="content_chunk", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
                FieldSchema(name="chunk_index", dtype=DataType.INT32),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            schema = CollectionSchema(fields, description="Car manual chunks")
            Collection(self.car_collection_name, schema)
        
        self.car_collection = Collection(self.car_collection_name)
        # Ensure index exists on the vector field
        if "vector" in [f.name for f in self.car_collection.schema.fields]:
            if not self.car_collection.indexes:
                self.car_collection.create_index(
                    field_name="vector",
                    index_params={
                        "index_type": "IVF_FLAT",
                        "metric_type": "L2",
                        "params": {"nlist": 128}
                    }
                )
        # Load the collection into memory
        self.car_collection.load()

    def insert(self, data: List[Dict[str, Any]]):
        # Data should be a list of dicts with keys matching the schema
        # SAFETY: Ensure all content_chunk values are within 8192 character limit
        valid_data = []
        for record in data:
            # Check content_chunk length
            content_chunk = record.get("content_chunk", "")
            if isinstance(content_chunk, str) and len(content_chunk) > 8192:
                record["content_chunk"] = content_chunk[:8192]
            
            # Check vector dimension (now 768)
            vector = record.get("vector", [])
            if isinstance(vector, list) and len(vector) == 768:
                valid_data.append(record)
            # Skip invalid records silently
        
        if not valid_data:
            return
        
        # FINAL ENFORCEMENT: Truncate all content_chunk fields in valid_data before insert
        for idx, record in enumerate(valid_data):
            chunk = record.get("content_chunk", "")
            if not isinstance(chunk, str):
                chunk = str(chunk)
            if len(chunk) > 8192:
                chunk = chunk[:8192]
            record["content_chunk"] = chunk
            if not isinstance(record["content_chunk"], str):
                raise ValueError(f"[MILVUS FINAL ENFORCEMENT ERROR] Chunk at index {idx} is not a string after conversion! Type: {type(record['content_chunk'])}")
            if len(record["content_chunk"]) > 8192:
                raise ValueError(f"[MILVUS FINAL ENFORCEMENT ERROR] Chunk at index {idx} still exceeds 8192 chars after truncation! Length: {len(record['content_chunk'])}. Preview: {record['content_chunk'][:100]!r}")
        
        def truncate_utf8_bytes(s, max_bytes):
            if not isinstance(s, str):
                s = str(s)
            b = s.encode('utf-8')
            if len(b) <= max_bytes:
                return s
            # Truncate bytes and decode safely
            truncated = b[:max_bytes]
            while True:
                try:
                    return truncated.decode('utf-8')
                except UnicodeDecodeError:
                    truncated = truncated[:-1]
        # Build insert_data with hard truncation for content_chunk (by bytes)
        fields = ["id", "book_title", "content_chunk", "vector", "page_number", "metadata"]
        insert_data = []
        for f in fields:
            if f == "content_chunk":
                col = [truncate_utf8_bytes(d.get(f, ""), 8192) for d in valid_data]
            else:
                col = [d.get(f) for d in valid_data]
            insert_data.append(col)
        self.collection.insert(insert_data)

        # Final truncation of content_chunk to ensure it does not exceed 8192 characters
        for record in data:
            if "content_chunk" in record and isinstance(record["content_chunk"], str):
                record["content_chunk"] = record["content_chunk"][:8192] if len(record["content_chunk"]) > 8192 else record["content_chunk"]

    def search(self, query_vector: List[float], top_k: int = 5, filter_expr: Optional[str] = None):
        # Ensure the collection is loaded before searching
        self.collection.load()
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
    
    def insert_car_manual(self, data: List[Dict[str, Any]]):
        """Insert car manual chunks into Milvus"""
        valid_data = []
        for record in data:
            # Check content_chunk length
            content_chunk = record.get("content_chunk", "")
            if isinstance(content_chunk, str) and len(content_chunk) > 8192:
                record["content_chunk"] = content_chunk[:8192]
            
            # Check vector dimension (768)
            vector = record.get("vector", [])
            if isinstance(vector, list) and len(vector) == 768:
                valid_data.append(record)
            # Skip invalid records silently
        
        if not valid_data:
            return
        
        # Ensure all required fields are present
        for idx, record in enumerate(valid_data):
            chunk = record.get("content_chunk", "")
            if not isinstance(chunk, str):
                chunk = str(chunk)
            if len(chunk) > 8192:
                chunk = chunk[:8192]
            record["content_chunk"] = chunk
        
        # Build insert_data for car collection
        fields = ["id", "car_id", "make", "model", "year", "content_chunk", "vector", "chunk_index", "source", "metadata"]
        insert_data = []
        for f in fields:
            col = [d.get(f) for d in valid_data]
            insert_data.append(col)
        
        # Ensure the collection is loaded before inserting
        self.car_collection.load()
        self.car_collection.insert(insert_data)

    def search_car_manual(self, query_vector: List[float], car_id: str = None, top_k: int = 5):
        """Search car manual chunks"""
        self.car_collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # Build filter expression
        filter_expr = 'source == "owner_manual"'
        if car_id:
            filter_expr += f' and car_id == "{car_id}"'
        
        results = self.car_collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["id", "car_id", "make", "model", "year", "content_chunk", "chunk_index", "metadata"]
        )
        return results

    def delete_car_manual_by_car_id(self, car_id: str):
        """Delete all chunks for a specific car"""
        # Ensure the collection is loaded before deleting
        self.car_collection.load()
        expr = f'car_id == "{car_id}"'
        self.car_collection.delete(expr)
