from fastapi import APIRouter, BackgroundTasks
from app.tasks import parse_pdf_task

router = APIRouter()

@router.post("/parse-pdf-background")
def parse_pdf_background(file_path: str, chunk_size: int = 1000):
    # Enqueue the Celery task
    task = parse_pdf_task.delay(file_path, chunk_size)
    return {"task_id": task.id, "status": "processing"}

@router.get("/task-status/{task_id}")
def get_task_status(task_id: str):
    from app.core.celery import celery_app
    result = celery_app.AsyncResult(task_id)
    return {"task_id": task_id, "status": result.status, "result": result.result if result.ready() else None}
