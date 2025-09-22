from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any
from uuid import uuid4
from app.services.chat_service import ChatService
from app.CRUD import ChatSessionCRUD
from app.utils.diagnosis_tree import DiagnosisTreeNode
from app.schemas.Chat_Session import ChatSession
from datetime import datetime
from app.services.user_service import user_service
from app.utils.text_sanitizer import sanitize_user_message
import re

router = APIRouter(
    tags=["Chat"],
    responses={
        404: {"description": "Resource not found."},
        400: {"description": "Bad request."},
        200: {"description": "Successful Response."},
        500: {"description": "Internal server error."}
    }
)

chat_service = ChatService()
chat_session_crud = ChatSessionCRUD()

# Enhanced Request Schemas with Validation
class ChatMessageRequest(BaseModel):
    user_id: str
    message: str
    context: Optional[Dict[str, Any]] = None
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or len(v) < 1 or len(v) > 100:
            raise ValueError('User ID must be between 1 and 100 characters')
        if not re.match(r'^[a-zA-Z0-9\-_]+$', v):
            raise ValueError('User ID contains invalid characters')
        return v
    
    @validator('message')
    def validate_message(cls, v):
        if not v or len(v) < 1 or len(v) > 5000:
            raise ValueError('Message must be between 1 and 5000 characters')
        # Basic sanitization - remove potential HTML/script tags
        if re.search(r'<[^>]+>', v):
            raise ValueError('HTML tags are not allowed in messages')
        if re.search(r'javascript:|data:|vbscript:', v, re.IGNORECASE):
            raise ValueError('Potentially dangerous URLs are not allowed')
        # Apply text sanitization (normalize "my car's" to "the car's", etc.)
        sanitized = sanitize_user_message(v.strip())
        return sanitized
    
    @validator('context')
    def validate_context(cls, v):
        if v is not None:
            # Limit context size and depth
            import json
            try:
                context_str = json.dumps(v)
                if len(context_str) > 10000:  # 10KB limit
                    raise ValueError('Context data is too large')
            except (TypeError, ValueError) as e:
                raise ValueError('Invalid context data format')
        return v

class ChatSessionMessageRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None
    
    @validator('message')
    def validate_message(cls, v):
        if not v or len(v) < 1 or len(v) > 5000:
            raise ValueError('Message must be between 1 and 5000 characters')
        if re.search(r'<[^>]+>', v):
            raise ValueError('HTML tags are not allowed in messages')
        if re.search(r'javascript:|data:|vbscript:', v, re.IGNORECASE):
            raise ValueError('Potentially dangerous URLs are not allowed')
        # Apply text sanitization (normalize "my car's" to "the car's", etc.)
        sanitized = sanitize_user_message(v.strip())
        return sanitized

class CreateChatSessionRequest(BaseModel):
    user_id: str
    context: Optional[Dict[str, Any]] = None
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or len(v) < 1 or len(v) > 100:
            raise ValueError('User ID must be between 1 and 100 characters')
        if not re.match(r'^[a-zA-Z0-9\-_]+$', v):
            raise ValueError('User ID contains invalid characters')
        return v
    
    @validator('context')
    def validate_context(cls, v):
        if v is not None:
            import json
            try:
                context_str = json.dumps(v)
                if len(context_str) > 10000:
                    raise ValueError('Context data is too large')
            except (TypeError, ValueError):
                raise ValueError('Invalid context data format')
        return v

# --- Session Management --
@router.post('/chat/session/create', summary="Create a new chat session")
async def create_chat_session(request: CreateChatSessionRequest):
    try:
        # Check if user_id exists
        if not await user_service.user_exists(request.user_id):
            raise HTTPException(status_code=404, detail="User ID does not exist")
        session_id = str(uuid4())
        now = datetime.now().isoformat()
        diagnosis_tree = DiagnosisTreeNode(issue_name='root', likelyhood=1.0)
        diagnosis_tree_dict = ChatSession.serialize_diagnosis_tree(diagnosis_tree)
        session_data = {
            'id': session_id,
            'user_id': request.user_id,
            'title': 'New Chat Session',
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

@router.get(
    '/chat/sessions/{user_id}',
    summary="Get all chat sessions for a user",
    description="Retrieve all chat sessions associated with a user. Returns only session metadata without messages.",
    responses={
        200: {"description": "List of chat sessions with metadata only."},
        404: {"description": "User not found."},
        500: {"description": "Internal server error."}
    }
)
async def get_user_sessions(user_id: str):
    try:
        sessions = await chat_session_crud.read({'user_id': user_id})
        # Return only session metadata, excluding messages to reduce logging verbosity
        session_summaries = []
        for session in sessions:
            summary = {
                'id': session.get('id'),
                'user_id': session.get('user_id'),
                'title': session.get('title', 'Untitled Session'),
                'created_at': session.get('created_at'),
                'updated_at': session.get('updated_at'),
                'message_count': len(session.get('messages', []))
            }
            session_summaries.append(summary)
        return session_summaries
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete(
    '/chat/session/{session_id}',
    summary="Delete a chat session",
    description="Delete a chat session by its session ID.",
    responses={
        200: {"description": "Session deleted."},
        404: {"description": "Session not found."},
        500: {"description": "Internal server error."}
    }
)
async def delete_session(session_id: str):
    try:
        result = await chat_session_crud.delete({'id': session_id})
        return {"message": "Session deleted", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Message Management ---
@router.get(
    '/chat/session/{session_id}/messages',
    summary="Get all messages in a session",
    description="Retrieve all messages from a specific chat session.",
    responses={
        200: {"description": "List of messages in the session."},
        404: {"description": "Session not found."},
        500: {"description": "Internal server error."}
    }
)
async def get_session_messages(session_id: str):
    try:
        sessions = await chat_session_crud.read({'id': session_id})
        if not sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return sessions[0].get('messages', [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete(
    '/chat/session/{session_id}/message/{message_index}',
    summary="Delete a specific message from a session",
    description="Delete a specific message from a chat session by its index.",
    responses={
        200: {"description": "Message deleted."},
        400: {"description": "Invalid message index."},
        404: {"description": "Session not found."},
        500: {"description": "Internal server error."}
    }
)
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
@router.post(
    '/chat/session/{session_id}/send',
    summary="Send a message within a session",
    description="Send a message within a chat session and receive the assistant's response.",
    responses={
        200: {"description": "Assistant response and updated messages."},
        404: {"description": "Session not found."},
        500: {"description": "Internal server error."}
    }
)
async def send_message_in_session(session_id: str, request: ChatSessionMessageRequest):
    try:
        # Try to find the session in the database
        sessions = await chat_session_crud.read({'id': session_id})
        
        if not sessions:
            # If session not found, provide a more detailed error message
            raise HTTPException(status_code=404, detail="Session not found")
            
        session = sessions[0]
        
        # Use chat_service to generate assistant response and update conversation
        response = await chat_service.send_message(
            user_id=session.get('user_id'),
            message=request.message,
            context=request.context,
            session=session
        )
        
        # Update the session in the DB with the new messages, updated_at and diagnosis_tree
        updated_session = chat_service._get_conversation(session.get('user_id'), session=session)
        
        # Serialize diagnosis tree for storage
        from app.schemas.Chat_Session import ChatSession
        diagnosis_tree_dict = None
        if 'context' in updated_session and updated_session['context'] and 'diagnosis_tree' in updated_session['context']:
            diagnosis_tree = updated_session['context'].get('diagnosis_tree')
            if diagnosis_tree:
                diagnosis_tree_dict = ChatSession.serialize_diagnosis_tree(diagnosis_tree)
            else:
                # Diagnosis tree is None in updated_session context
                pass
        else:
            # No diagnosis tree found in updated_session context
            pass
        
        # If we don't have a diagnosis tree, create a default one to satisfy not-null constraint
        if diagnosis_tree_dict is None:
            from app.utils.diagnosis_tree import DiagnosisTreeNode
            default_tree = DiagnosisTreeNode(issue_name='root', likelyhood=1.0)
            diagnosis_tree_dict = ChatSession.serialize_diagnosis_tree(default_tree)
        
        # Ensure updated_at is always a string
        updated_at_str = updated_session.get('last_updated')
        if not isinstance(updated_at_str, str):
            updated_at_str = datetime.now().isoformat()
            
        await chat_session_crud.update({'id': session_id}, {
            'title': updated_session.get('title', 'Chat Session'),
            'messages': updated_session['messages'],
            'updated_at': updated_at_str,
            'diagnosis_tree': diagnosis_tree_dict
        })
        return {"messages": updated_session['messages']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- General Chat (no session) ---
@router.post('/chat/send', summary="Send a chat message", response_description="Chat assistant response",
    description="""
    Send a message to the chat assistant and receive a contextual response.
    - **user_id**: Unique identifier for the user
    - **message**: The user's message
    - **context**: (Optional) Additional context such as car details or symptoms
    Returns a response with answer, confidence, sources, and metadata.
    """)
async def send_chat_message(request: ChatMessageRequest):
    try:
        # Check if user_id exists
        if not await user_service.user_exists(request.user_id):
            raise HTTPException(status_code=404, detail="User ID does not exist")
        # Create a new session for each request
        session_id = str(uuid4())
        now = datetime.now().isoformat()
        session_data = {
            'id': session_id,
            'user_id': request.user_id,
            'messages': [],
            'created_at': now,
            'updated_at': now,
            'context': request.context or {}
        }
        # Optionally, persist the session if needed:
        # await chat_session_crud.create(session_data)
        # Pass the context to chat_service.send_message
        response = await chat_service.send_message(
            user_id=request.user_id,
            message=request.message,
            context=request.context
        )
        # Optionally, return the session_id in the response
        response['session_id'] = session_id
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Performance ---
@router.get('/chat/performance', summary="Get chat performance stats", response_description="Performance statistics",
    description="""
    Get performance statistics for the chat service, including average, min, and max response times, and conversation counts.
    """)
def get_performance_stats():
    return chat_service.get_performance_stats()
