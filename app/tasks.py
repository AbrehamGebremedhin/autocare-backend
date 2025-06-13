from app.core.celery import celery_app
from app.services.parser_service import ParserService

@celery_app.task
def parse_pdf_task(file_path: str, chunk_size: int = 1000):
    parser = ParserService()
    # Celery tasks should not be async, so we run the async method in an event loop
    import asyncio
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(parser.parse_pdf(file_path, chunk_size))
