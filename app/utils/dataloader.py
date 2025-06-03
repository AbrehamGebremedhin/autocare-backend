import os
import asyncio
import inspect
import traceback
import uuid
import time
from app.services.parser_service import ParserService
from app.services.embedding_service import EmbeddingService
from app.db.milvus_handler import MilvusHandler
from app.utils.logger import Logger

CAR_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'car_data')
TABLE_NAME = "Groundknowledge"

MAX_EMBEDDING_RETRIES = 3
MAX_INSERT_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds

async def process_pdf(pdf_path, parser_service, embedding_service, milvus_handler, logger):
    book_title = os.path.splitext(os.path.basename(pdf_path))[0]
    try:
        await logger.info(f"Processing PDF: {pdf_path}")
        chunks = await parser_service.parse_pdf(pdf_path)
        if not chunks:
            await logger.warning(f"No text extracted from {pdf_path}")
            return
        # Build a list of safe, truncated strings for both embedding and insertion
        def sanitize_chunk(chunk):
            chunk_str = str(chunk)
            if len(chunk_str) > 8192:
                chunk_str = chunk_str[:8192]
            return chunk_str
        safe_chunks = [sanitize_chunk(chunk) for chunk in chunks]
        # Retry embedding with exponential backoff
        for attempt in range(MAX_EMBEDDING_RETRIES):
            try:
                embeddings_result = embedding_service.embed_texts(safe_chunks)
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
                    return
        # Prepare data for Milvus, with strict enforcement of chunk length
        milvus_data = []
        for idx, (chunk_str, vector) in enumerate(zip(safe_chunks, embeddings)):
            # chunk_str is already sanitized, but double-check defensively
            if len(chunk_str) > 8192:
                await logger.warning(f"[CRITICAL] Chunk at index {idx} still exceeds 8192 chars (length: {len(chunk_str)}). Truncating before insertion.")
                chunk_str = chunk_str[:8192]
            assert len(chunk_str) <= 8192, f"Chunk at index {idx} exceeds 8192 chars after sanitization!"
            data = {
                "id": str(uuid.uuid4()),
                "book_title": book_title,
                "content_chunk": chunk_str,
                "vector": vector,
                "page_number": -1,  # Use -1 if page number is unknown
                "metadata": {"source_file": pdf_path, "chunk_index": idx}
            }
            milvus_data.append(data)
        # Insert all at once (Milvus is batch-friendly)
        # FINAL ENFORCEMENT: Force string and truncate all content_chunk fields before insert
        for idx, record in enumerate(milvus_data):
            chunk = record["content_chunk"]
            if not isinstance(chunk, str):
                await logger.warning(f"[FINAL ENFORCEMENT] Chunk at index {idx} is not a string (type: {type(chunk)}). Repr: {repr(chunk)[:100]!r}")
                chunk = str(chunk)
            # Ensure content_chunk is truncated to 8192 characters
            record["content_chunk"] = chunk[:8192] if len(chunk) > 8192 else chunk
            if not isinstance(record["content_chunk"], str):
                await logger.error(f"[FINAL ENFORCEMENT ERROR] Chunk at index {idx} is not a string after conversion! Type: {type(record['content_chunk'])}")
                raise ValueError(f"[FINAL ENFORCEMENT ERROR] Chunk at index {idx} is not a string after conversion!")
            if len(record["content_chunk"]) > 8192:
                await logger.error(f"[FINAL ENFORCEMENT ERROR] Chunk at index {idx} still exceeds 8192 chars after truncation! Length: {len(record['content_chunk'])}. Preview: {record['content_chunk'][:100]!r}")
                raise ValueError(f"[FINAL ENFORCEMENT ERROR] Chunk at index {idx} still exceeds 8192 chars after truncation!")
        # FINAL DIAGNOSTIC: Check all content_chunk fields before insert
        for idx, record in enumerate(milvus_data):
            chunk = record["content_chunk"]
            if len(chunk) > 8192:
                await logger.error(f"[DIAGNOSTIC ERROR] About to insert overlong chunk at index {idx} (length: {len(chunk)}). Preview: {chunk[:100]!r}")
                raise ValueError(f"[DIAGNOSTIC ERROR] About to insert overlong chunk at index {idx} (length: {len(chunk)})")
        for attempt in range(MAX_INSERT_RETRIES):
            try:
                await logger.info(f"Inserting {len(milvus_data)} records into Milvus collection '{TABLE_NAME}'")
                milvus_handler.insert(milvus_data)
                break  # Success
            except Exception as e:
                if attempt < MAX_INSERT_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    await logger.warning(f"Milvus insert failed (attempt {attempt+1}/{MAX_INSERT_RETRIES}): {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    await logger.error(f"Milvus insert failed after {MAX_INSERT_RETRIES} attempts: {e}")
        await logger.info(f"Finished processing {pdf_path}")
    except Exception as e:
        tb = traceback.format_exc()
        await logger.error(f"Error processing {pdf_path}: {repr(e)}\n{tb}")

async def main():
    logger = Logger("dataloader")
    parser_service = ParserService()
    embedding_service = EmbeddingService()
    milvus_handler = MilvusHandler(collection_name=TABLE_NAME)

    pdf_files = [os.path.join(CAR_DATA_DIR, f) for f in os.listdir(CAR_DATA_DIR) if f.lower().endswith('.pdf')]
    await logger.info(f"Found {len(pdf_files)} PDF files in {CAR_DATA_DIR}")
    tasks = [process_pdf(pdf, parser_service, embedding_service, milvus_handler, logger) for pdf in pdf_files]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
