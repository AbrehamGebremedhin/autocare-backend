from app.services.base_service import BaseService
from app.services.parser_service import ParserService
from app.services.embedding_service import EmbeddingService
from app.db.milvus_handler import MilvusHandler
from app.db.bucket_operations import SupabaseBucketManager
from app.utils.logger import get_logger_instance
from typing import List, Dict, Any, Optional
import uuid
import asyncio


class CarVectorizationService(BaseService):
    """Service for vectorizing car manual text and storing in Milvus"""
    
    def __init__(
        self,
        websocket_manager=None,
        parser_service: Optional[ParserService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        milvus_handler: Optional[MilvusHandler] = None,
        bucket_manager: Optional[SupabaseBucketManager] = None
    ):
        super().__init__(websocket_manager=websocket_manager)
        self.parser_service = parser_service or ParserService(websocket_manager=websocket_manager)
        self.embedding_service = embedding_service or EmbeddingService(websocket_manager=websocket_manager)
        self.milvus_handler = milvus_handler or MilvusHandler()
        self.bucket_manager = bucket_manager or SupabaseBucketManager()
        self.logger = get_logger_instance("car_vectorization")
    
    async def perform_action(self, action: str, *args, **kwargs):
        """
        Perform action for BaseService compatibility
        """
        if action == "vectorize":
            return await self.vectorize_car_manual_text(*args, **kwargs)
        elif action == "search":
            return await self.search_car_manual(*args, **kwargs)
        elif action == "status":
            return await self.get_car_vectorization_status(*args, **kwargs)
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
        
    async def vectorize_car_manual_text(
        self, 
        car_id: str, 
        make: str, 
        model: str, 
        year: int, 
        manual_text: str,
        websocket=None,
        session_id: str = None,
        chunk_size: int = 800,
        overlap: int = 200
    ) -> Dict[str, Any]:
        """
        Vectorize car manual text with configurable chunking and overlap and store in Milvus
        
        Args:
            car_id: Unique car identifier
            make: Car make
            model: Car model  
            year: Car year
            manual_text: Full manual text to vectorize
            websocket: Optional websocket for progress updates
            session_id: Optional session ID for websocket messages
            chunk_size: Size of text chunks (default: 800)
            overlap: Overlap between chunks (default: 200)
            
        Returns:
            Dict with success status and chunk count
        """
        try:
            if not manual_text or not manual_text.strip():
                return {"success": False, "error": "No manual text provided", "chunk_count": 0}
                
            await self.logger.info(f"Starting vectorization for car {car_id} ({make} {model} {year}) with chunk_size={chunk_size}, overlap={overlap}")
            
            if websocket:
                await self.send_ws_progress(
                    websocket, 
                    f"Starting vectorization for {make} {model} {year}",
                    self.__class__.__name__,
                    0.1,
                    session_id=session_id
                )
            
            # Delete existing chunks for this car
            await self.logger.info(f"Cleaning existing chunks for car {car_id}")
            self.milvus_handler.delete_car_manual_by_car_id(car_id)
            
            # Chunk the text with overlap
            await self.logger.info(f"Chunking manual text for car {car_id} (chunk_size={chunk_size}, overlap={overlap})")
            chunks = await self._chunk_text_with_overlap(manual_text, chunk_size, overlap)
            
            if not chunks:
                return {"success": False, "error": "No chunks generated from manual text", "chunk_count": 0}
            
            await self.logger.info(f"Generated {len(chunks)} chunks for car {car_id}")
            
            if websocket:
                await self.send_ws_progress(
                    websocket,
                    f"Generated {len(chunks)} text chunks",
                    self.__class__.__name__, 
                    0.3,
                    session_id=session_id
                )
            
            # Generate embeddings for chunks
            await self.logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embeddings = await self.embedding_service.embed_texts_batch(chunks)
            
            if len(embeddings) != len(chunks):
                return {"success": False, "error": f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)}", "chunk_count": 0}
            
            if websocket:
                await self.send_ws_progress(
                    websocket,
                    f"Generated embeddings for {len(chunks)} chunks", 
                    self.__class__.__name__,
                    0.7,
                    session_id=session_id
                )
            
            # Prepare data for Milvus insertion
            milvus_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{car_id}-chunk-{i:04d}"
                
                record = {
                    "id": chunk_id,
                    "car_id": car_id,
                    "make": make.lower(),
                    "model": model.lower(), 
                    "year": year,
                    "content_chunk": chunk[:8192],  # Ensure chunk length limit
                    "vector": embedding,
                    "chunk_index": i,
                    "source": "owner_manual",
                    "metadata": {
                        "car_id": car_id,
                        "make": make,
                        "model": model,
                        "year": year,
                        "chunk_length": len(chunk)
                    }
                }
                milvus_data.append(record)
            
            # Insert into Milvus
            await self.logger.info(f"Inserting {len(milvus_data)} records into Milvus for car {car_id}")
            self.milvus_handler.insert_car_manual(milvus_data)
            
            if websocket:
                await self.send_ws_result(
                    websocket,
                    f"Successfully vectorized {len(chunks)} chunks for {make} {model} {year}",
                    self.__class__.__name__,
                    session_id=session_id,
                    details={"chunk_count": len(chunks), "car_id": car_id}
                )
            
            await self.logger.info(f"Successfully vectorized car {car_id} with {len(chunks)} chunks")
            return {"success": True, "chunk_count": len(chunks)}
            
        except Exception as e:
            error_msg = f"Error vectorizing car {car_id}: {str(e)}"
            await self.logger.error(error_msg)
            
            if websocket:
                await self.send_ws_error(
                    websocket,
                    error_msg,
                    self.__class__.__name__,
                    session_id=session_id,
                    details={"car_id": car_id, "error": str(e)}
                )
                
            return {"success": False, "error": str(e), "chunk_count": 0}

    async def vectorize_car_manual_from_pdf(
        self,
        car_id: str,
        make: str, 
        model: str,
        year: int,
        pdf_filename: str,
        websocket=None,
        session_id: str = None,
        chunk_size: int = 800,
        overlap: int = 200
    ) -> Dict[str, Any]:
        """
        Extract text from PDF and vectorize it with configurable chunking
        
        Args:
            car_id: Unique car identifier
            make: Car make
            model: Car model
            year: Car year
            pdf_filename: Name of PDF file in bucket
            websocket: Optional websocket for progress updates
            session_id: Optional session ID for websocket messages
            chunk_size: Size of text chunks (default: 800)
            overlap: Overlap between chunks (default: 200)
            
        Returns:
            Dict with success status and chunk count
        """
        try:
            bucket_name = "manuals"
            
            # Download PDF from bucket
            await self.logger.info(f"Downloading PDF {pdf_filename} for car {car_id}")
            file_bytes = await self.bucket_manager.download_file(bucket_name, pdf_filename)
            
            if not file_bytes:
                return {"success": False, "error": f"Could not download PDF {pdf_filename}", "chunk_count": 0}
            
            if websocket:
                await self.send_ws_progress(
                    websocket,
                    f"Downloaded PDF for {make} {model} {year}",
                    self.__class__.__name__,
                    0.2,
                    session_id=session_id
                )
            
            # Extract text from PDF
            await self.logger.info(f"Extracting text from PDF for car {car_id}")
            chunks = await self.parser_service.parse_pdf_bytes_optimized(file_bytes)
            
            if not chunks:
                return {"success": False, "error": "Could not extract text from PDF", "chunk_count": 0}
            
            manual_text = '\n'.join(chunks)
            
            # Vectorize the extracted text
            return await self.vectorize_car_manual_text(
                car_id=car_id,
                make=make,
                model=model,
                year=year,
                manual_text=manual_text,
                websocket=websocket,
                session_id=session_id,
                chunk_size=chunk_size,
                overlap=overlap
            )
            
        except Exception as e:
            error_msg = f"Error vectorizing PDF for car {car_id}: {str(e)}"
            await self.logger.error(error_msg)
            
            if websocket:
                await self.send_ws_error(
                    websocket,
                    error_msg,
                    self.__class__.__name__,
                    session_id=session_id,
                    details={"car_id": car_id, "error": str(e)}
                )
                
            return {"success": False, "error": str(e), "chunk_count": 0}

    async def search_car_manual(
        self, 
        query: str, 
        car_id: str = None, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search car manual chunks in Milvus
        
        Args:
            query: Search query
            car_id: Optional car ID to filter by
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_text(query)
            
            # Search Milvus
            results = self.milvus_handler.search_car_manual(
                query_vector=query_embedding,
                car_id=car_id,
                top_k=top_k
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "id": hit.entity.get("id", ""),
                        "car_id": hit.entity.get("car_id", ""),
                        "make": hit.entity.get("make", ""),
                        "model": hit.entity.get("model", ""),
                        "year": hit.entity.get("year", 0),
                        "chunk": hit.entity.get("content_chunk", ""),
                        "chunk_index": hit.entity.get("chunk_index", 0),
                        "score": float(hit.distance),
                        "source": "owner_manual",
                        "metadata": hit.entity.get("metadata", {})
                    })
            
            return formatted_results
            
        except Exception as e:
            await self.logger.error(f"Error searching car manual: {str(e)}")
            return []

    async def _chunk_text_with_overlap(self, text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
        """
        Chunk text with configurable overlap between chunks
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks with overlap
        """
        if not text or chunk_size <= 0:
            return []
        
        # Ensure overlap is not larger than chunk_size
        overlap = min(overlap, chunk_size - 1)
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)
            
            # Move start position with overlap consideration
            start = end - overlap
            
            # If we're at the end, break to avoid infinite loop
            if end >= len(text):
                break
        
        return chunks

    async def get_car_vectorization_status(self, car_id: str) -> Dict[str, Any]:
        """
        Check if car manual is vectorized and get chunk count
        
        Args:
            car_id: Car identifier
            
        Returns:
            Dict with vectorization status and chunk count
        """
        try:
            # Query Milvus for car chunks
            results = self.milvus_handler.car_collection.query(
                expr=f'car_id == "{car_id}"',
                output_fields=["id", "chunk_index"],
                limit=10000  # High limit to count all chunks
            )
            
            chunk_count = len(results)
            is_vectorized = chunk_count > 0
            
            return {
                "is_vectorized": is_vectorized,
                "chunk_count": chunk_count
            }
            
        except Exception as e:
            await self.logger.error(f"Error checking vectorization status for car {car_id}: {str(e)}")
            return {"is_vectorized": False, "chunk_count": 0}
