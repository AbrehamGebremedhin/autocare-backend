import os
import asyncio
from app.services.parser_service import ParserService
from app.services.embedding_service import EmbeddingService
from app.db.base import SupabaseDBHandler
from app.utils.logger import Logger

CAR_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'car_data')
TABLE_NAME = "Ground_Knowledge"

async def process_pdf(pdf_path, parser_service, embedding_service, supabase_client, logger):
    book_title = os.path.splitext(os.path.basename(pdf_path))[0]
    try:
        await logger.info(f"Processing PDF: {pdf_path}")
        chunks = await parser_service.parse_pdf(pdf_path)
        if not chunks:
            await logger.warning(f"No text extracted from {pdf_path}")
            return
        embeddings = await embedding_service.embed_texts(chunks)
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            data = {
                "book_title": book_title,
                "content_chunk": chunk,
                "vector": vector,
                "page_number": None,  # Optional: can be improved if page info is available
                "metadata": {"source_file": pdf_path, "chunk_index": idx}
            }
            # Insert into Supabase
            supabase_client.table(TABLE_NAME).insert(data).execute()
        await logger.info(f"Finished processing {pdf_path}")
    except Exception as e:
        await logger.error(f"Error processing {pdf_path}: {e}")

async def main():
    logger = Logger("dataloader")
    parser_service = ParserService()
    embedding_service = EmbeddingService()
    db_handler = SupabaseDBHandler()
    supabase_client = await db_handler.client

    pdf_files = [os.path.join(CAR_DATA_DIR, f) for f in os.listdir(CAR_DATA_DIR) if f.lower().endswith('.pdf')]
    await logger.info(f"Found {len(pdf_files)} PDF files in {CAR_DATA_DIR}")
    tasks = [process_pdf(pdf, parser_service, embedding_service, supabase_client, logger) for pdf in pdf_files]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
