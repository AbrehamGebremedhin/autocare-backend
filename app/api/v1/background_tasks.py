from fastapi import APIRouter, BackgroundTasks
from app.tasks import parse_pdf_task

router = APIRouter()

@router.get("/task-status/{task_id}")
def get_task_status(task_id: str):
    from app.core.celery import celery_app
    result = celery_app.AsyncResult(task_id)
    return {"task_id": task_id, "status": result.status, "result": result.result if result.ready() else None}
