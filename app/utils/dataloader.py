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
        # Retry embedding with exponential backoff
        for attempt in range(MAX_EMBEDDING_RETRIES):
            try:
                embeddings_result = embedding_service.embed_texts(chunks)
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
        # Prepare data for Milvus
        milvus_data = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            data = {
                "id": str(uuid.uuid4()),
                "book_title": book_title,
                "content_chunk": chunk,
                "vector": vector,
                "page_number": -1,  # Use -1 if page number is unknown
                "metadata": {"source_file": pdf_path, "chunk_index": idx}
            }
            milvus_data.append(data)
        # Insert all at once (Milvus is batch-friendly)
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
