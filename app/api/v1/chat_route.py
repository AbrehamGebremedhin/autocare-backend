from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()

class ChatMessageRequest(BaseModel):
    user_id: str
    message: str
    context: Optional[Dict[str, Any]] = None

@router.post('/chat/send', summary="Send a chat message", response_description="Chat assistant response", tags=["Chat"],
    description="""
    Send a message to the chat assistant and receive a contextual response.
    - **user_id**: Unique identifier for the user
    - **message**: The user's message
    - **context**: (Optional) Additional context such as car details or symptoms
    Returns a response with answer, confidence, sources, and metadata.
    """)
async def send_chat_message(request: ChatMessageRequest):
    try:
        response = await chat_service.send_message(
            user_id=request.user_id,
            message=request.message,
            context=request.context
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/chat/history/{user_id}', summary="Get chat history", response_description="User's chat history", tags=["Chat"],
    description="""
    Retrieve the chat history for a user.
    - **user_id**: Unique identifier for the user
    - **limit**: Maximum number of messages to return (default: 50)
    Returns a list of messages and conversation metadata.
    """)
async def get_chat_history(user_id: str, limit: int = 50):
    try:
        history = await chat_service.get_history(user_id=user_id, limit=limit)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/chat/clear/{user_id}', summary="Clear chat history", response_description="Confirmation of cleared conversation", tags=["Chat"],
    description="""
    Clear the chat history for a user.
    - **user_id**: Unique identifier for the user
    Returns a confirmation message.
    """)
async def clear_chat_history(user_id: str):
    try:
        result = await chat_service.clear_conversation(user_id)
        if result:
            return {"message": "Conversation cleared."}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear conversation.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/chat/performance', summary="Get chat performance stats", response_description="Performance statistics", tags=["Chat"],
    description="""
    Get performance statistics for the chat service, including average, min, and max response times, and conversation counts.
    """)
def get_performance_stats():
    return chat_service.get_performance_stats()
