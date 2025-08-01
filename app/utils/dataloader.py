import os
import asyncio
import inspect
import traceback
import uuid
import time
import chardet
from typing import AsyncGenerator, List, Dict, Any, Optional
from app.services.parser_service import ParserService
from app.services.embedding_service import EmbeddingService
from app.db.milvus_handler import MilvusHandler
from app.utils.logger import get_logger_instance
from pymilvus import utility

CAR_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'car_data')
TABLE_NAME = "Groundknowledge"

MAX_EMBEDDING_RETRIES = 3
MAX_INSERT_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming
MAX_CHUNK_LENGTH = 8192  # Maximum characters per chunk

class DataValidationPipeline:
    """Pipeline for validating and preprocessing data"""
    
    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """Detect file encoding using chardet"""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    
    @staticmethod
    def sanitize_chunk(chunk: str, max_length: int = MAX_CHUNK_LENGTH) -> str:
        """Sanitize and validate text chunks"""
        if not isinstance(chunk, str):
            chunk = str(chunk)
        
        # Remove null bytes and other problematic characters
        chunk = chunk.replace('\x00', '').replace('\ufffd', '')
        
        # Normalize whitespace
        chunk = ' '.join(chunk.split())
        
        # Truncate if too long
        if len(chunk) > max_length:
            chunk = chunk[:max_length]
        
        return chunk.strip()
    
    @staticmethod
    def validate_chunks(chunks: List[str]) -> List[str]:
        """Validate a list of chunks"""
        validated_chunks = []
        for chunk in chunks:
            sanitized = DataValidationPipeline.sanitize_chunk(chunk)
            if sanitized and len(sanitized) > 10:  # Minimum length requirement
                validated_chunks.append(sanitized)
        return validated_chunks

class StreamingProcessor:
    """Handles streaming processing of large files"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE):
        self.chunk_size = chunk_size
    
    async def stream_file_chunks(self, file_path: str, encoding: str = 'utf-8') -> AsyncGenerator[str, None]:
        """Stream file content in chunks to handle large files"""
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as file:
                while True:
                    chunk = file.read(self.chunk_size)
                    if not chunk:
                        break
                    yield chunk
                    await asyncio.sleep(0)  # Yield control
        except Exception as e:
            raise IOError(f"Error streaming file {file_path}: {str(e)}")
    
    async def process_chunks_in_batches(self, chunks: List[str], batch_size: int = 100) -> AsyncGenerator[List[str], None]:
        """Process chunks in batches for memory efficiency"""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            yield batch
            await asyncio.sleep(0)  # Yield control

async def process_pdf_with_streaming(pdf_path, parser_service, embedding_service, milvus_handler, logger):
    """Enhanced PDF processing with streaming and validation"""
    book_title = os.path.splitext(os.path.basename(pdf_path))[0]
    
    try:
        await logger.info(f"Processing PDF with streaming: {pdf_path}")
        
        # Detect encoding for better character handling
        validation_pipeline = DataValidationPipeline()
        encoding = validation_pipeline.detect_encoding(pdf_path)
        await logger.info(f"Detected encoding: {encoding}")
        
        # Parse PDF with improved error handling
        chunks = await parser_service.parse_pdf(pdf_path)
        if not chunks:
            await logger.warning(f"No text extracted from {pdf_path}")
            return
        
        # Validate and sanitize chunks
        validated_chunks = validation_pipeline.validate_chunks(chunks)
        if not validated_chunks:
            await logger.warning(f"No valid chunks after validation for {pdf_path}")
            return
        
        await logger.info(f"Validated {len(validated_chunks)} chunks from {len(chunks)} original chunks")
        
        # Process in streaming batches
        streaming_processor = StreamingProcessor()
        total_processed = 0
        
        async for batch in streaming_processor.process_chunks_in_batches(validated_chunks, batch_size=50):
            # Retry embedding with exponential backoff
            embeddings = None
            for attempt in range(MAX_EMBEDDING_RETRIES):
                try:
                    embeddings_result = embedding_service.embed_texts(batch)
                    if inspect.isawaitable(embeddings_result):
                        embeddings = await embeddings_result
                    else:
                        embeddings = embeddings_result
                    break  # Success
                except Exception as e:
                    if attempt < MAX_EMBEDDING_RETRIES - 1:
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        await logger.warning(f"Embedding failed (attempt {attempt+1}/{MAX_EMBEDDING_RETRIES}): {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        await logger.error(f"Embedding failed after {MAX_EMBEDDING_RETRIES} attempts: {e}")
                        continue  # Skip this batch
            
            if embeddings is None:
                await logger.error(f"Skipping batch due to embedding failures")
                continue
            
            # Prepare data for insertion with enhanced metadata
            data_to_insert = []
            for i, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                entry = {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "embedding": embedding,
                    "source": book_title,
                    "chunk_index": total_processed + i,
                    "file_path": pdf_path,
                    "encoding": encoding,
                    "processed_at": time.time()
                }
                data_to_insert.append(entry)
            
            # Insert with retry logic
            for attempt in range(MAX_INSERT_RETRIES):
                try:
                    await milvus_handler.insert(TABLE_NAME, data_to_insert)
                    break  # Success
                except Exception as e:
                    if attempt < MAX_INSERT_RETRIES - 1:
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        await logger.warning(f"Insert failed (attempt {attempt+1}/{MAX_INSERT_RETRIES}): {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        await logger.error(f"Insert failed after {MAX_INSERT_RETRIES} attempts: {e}")
                        break
            
            total_processed += len(batch)
            await logger.info(f"Processed {total_processed}/{len(validated_chunks)} chunks from {book_title}")
        
        await logger.info(f"Successfully processed {total_processed} chunks from {pdf_path}")
        
    except Exception as e:
        await logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        await logger.error(f"Traceback: {traceback.format_exc()}")

async def process_pdf(pdf_path, parser_service, embedding_service, milvus_handler, logger):
    """Legacy function that now calls the enhanced streaming version"""
    return await process_pdf_with_streaming(pdf_path, parser_service, embedding_service, milvus_handler, logger)

async def main():
    logger = get_logger_instance("dataloader")
    parser_service = ParserService()
    embedding_service = EmbeddingService()
    milvus_handler = MilvusHandler(collection_name=TABLE_NAME)

    # Check if the collection exists and how many documents it has
    try:
        collection_exists = utility.has_collection(TABLE_NAME)
    except Exception as e:
        await logger.error(f"Error checking if collection exists: {e}")
        collection_exists = False

    if collection_exists:
        try:
            count = milvus_handler.count_documents() if hasattr(milvus_handler, 'count_documents') else milvus_handler.count()
        except Exception as e:
            await logger.error(f"Error counting documents in collection: {e}")
            count = 0
        if count > 27000:
            await logger.info(f"Database already loaded with {count} documents. Skipping loading.")
            return
        else:
            await logger.info(f"Database exists but only has {count} documents. Proceeding to load.")
    else:
        await logger.info(f"Collection '{TABLE_NAME}' does not exist. Proceeding to load.")

    pdf_files = [os.path.join(CAR_DATA_DIR, f) for f in os.listdir(CAR_DATA_DIR) if f.lower().endswith('.pdf')]
    await logger.info(f"Found {len(pdf_files)} PDF files in {CAR_DATA_DIR}")
    
    # Process files with enhanced streaming and validation
    tasks = [process_pdf_with_streaming(pdf, parser_service, embedding_service, milvus_handler, logger) for pdf in pdf_files]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
