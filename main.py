from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.api.v1.routes import router as v1_router
from app.utils.websocket import manager

app = FastAPI()

app.include_router(v1_router, prefix="/api/v1")

@app.get("/")
async def read_root():
    return {"message": "Welcome to AutoCare API"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"You wrote: {data}", websocket)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)