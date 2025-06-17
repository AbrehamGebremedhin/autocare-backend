from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Optional, Dict, Any
from uuid import uuid4
from app.services.chat_service import ChatService
from app.CRUD import ChatSessionCRUD
from app.utils.diagnosis_tree import DiagnosisTreeNode
from app.schemas.Chat_Session import ChatSession
from datetime import datetime

router = APIRouter()
chat_service = ChatService()
chat_session_crud = ChatSessionCRUD()

# --- Request Schemas ---
class ChatMessageRequest(BaseModel):
    user_id: str
    message: str
    context: Optional[Dict[str, Any]] = None

class ChatSessionMessageRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class CreateChatSessionRequest(BaseModel):
    user_id: str
    context: Optional[Dict[str, Any]] = None

# --- Session Management --
@router.post('/chat/session/create', summary="Create a new chat session", tags=["Chat"])
async def create_chat_session(request: CreateChatSessionRequest):
    try:
        session_id = str(uuid4())
        now = datetime.now().isoformat()
        diagnosis_tree = DiagnosisTreeNode(issue_name='root', likelyhood=1.0)
        diagnosis_tree_dict = ChatSession.serialize_diagnosis_tree(diagnosis_tree)
        session_data = {
            'id': session_id,
            'user_id': request.user_id,
            'messages': [],
            'created_at': now,
            'updated_at': now,
            'context': request.context or {},
            'diagnosis_tree': diagnosis_tree_dict
        }
        await chat_session_crud.create(session_data)
        return session_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/chat/sessions/{user_id}', summary="Get all chat sessions for a user", tags=["Chat"])
async def get_user_sessions(user_id: str):
    try:
        sessions = await chat_session_crud.read({'user_id': user_id})
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete('/chat/session/{session_id}', summary="Delete a chat session", tags=["Chat"])
async def delete_session(session_id: str):
    try:
        result = await chat_session_crud.delete({'id': session_id})
        return {"message": "Session deleted", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Message Management ---
@router.get('/chat/session/{session_id}/messages', summary="Get all messages in a session", tags=["Chat"])
async def get_session_messages(session_id: str):
    try:
        sessions = await chat_session_crud.read({'id': session_id})
        if not sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return sessions[0].get('messages', [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete('/chat/session/{session_id}/message/{message_index}', summary="Delete a specific message from a session", tags=["Chat"])
async def delete_message_from_session(session_id: str, message_index: int):
    try:
        sessions = await chat_session_crud.read({'id': session_id})
        if not sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        session = sessions[0]
        messages = session.get('messages', [])
        if message_index < 0 or message_index >= len(messages):
            raise HTTPException(status_code=400, detail="Invalid message index")
        messages.pop(message_index)
        session['messages'] = messages
        session['updated_at'] = datetime.now().isoformat()
        await chat_session_crud.update({'id': session_id}, {'messages': messages, 'updated_at': session['updated_at']})
        return {"message": "Message deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Chatting ---
@router.post('/chat/session/{session_id}/send', summary="Send a message within a session", tags=["Chat"])
async def send_message_in_session(session_id: str, request: ChatSessionMessageRequest):
    try:
        sessions = await chat_session_crud.read({'id': session_id})
        if not sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        session = sessions[0]
        messages = session.get('messages', [])
        user_message = {
            'role': 'user',
            'content': request.message,
            'timestamp': datetime.now().isoformat(),
            'context': request.context,
            'is_initial': len(messages) == 0
        }
        messages.append(user_message)
        # Optionally, generate assistant response (reuse chat_service logic if needed)
        assistant_message = {
            'role': 'assistant',
            'content': f"Echo: {request.message}",
            'timestamp': datetime.now().isoformat(),
            'confidence': 1.0,
            'sources': []
        }
        messages.append(assistant_message)
        session['messages'] = messages
        session['updated_at'] = datetime.now().isoformat()
        await chat_session_crud.update({'id': session_id}, {'messages': messages, 'updated_at': session['updated_at']})
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- General Chat (no session) ---
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

# --- Performance ---
@router.get('/chat/performance', summary="Get chat performance stats", response_description="Performance statistics", tags=["Chat"],
    description="""
    Get performance statistics for the chat service, including average, min, and max response times, and conversation counts.
    """)
def get_performance_stats():
    return chat_service.get_performance_stats()
